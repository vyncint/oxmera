//! `CudaBackend`: one context, one stream, one module of eight kernels.
//!
//! Every launch goes through [`CudaBackend::launch`], which pairs a kernel
//! name with the argument list its CUDA C signature expects; the
//! signatures are in `kernels.cu` next to this file.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchArgs,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{DType, Device, Error, Layout, Result, Shape};
use oxmera_tensor::backend::{
    AdamStep, Backend, BinaryOp, MatmulPlan, ReduceOp, UnaryOp, plan_matmul, register_backend,
};
use oxmera_tensor::storage::{OpaqueBuffer, Storage};
use oxmera_tensor::tensor::Tensor;

/// The CUDA C source, embedded so a driver that rejects the shipped PTX
/// can rebuild it through NVRTC.
const KERNELS_CU: &str = include_str!("../kernels.cu");
/// PTX for `compute_75`, produced by `nvcc -arch=compute_75 -O3 -ptx`.
const KERNELS_PTX: &str = include_str!("../kernels.ptx");
const KERNEL_NAMES: &[&str] = &[
    "unary_strided",
    "binary_strided",
    "reduce_axis",
    "reduce_full_partials",
    "matmul_tiled",
    "gather_dim",
    "scatter_add_dim",
    "adam_step",
];

/// Host mirror of `struct AdamArgs` in `kernels.cu` (ten four-byte words).
#[repr(C)]
#[derive(Clone, Copy)]
struct AdamArgs {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32,
    bc2: f32,
    decoupled: u32,
    has_state: u32,
    numel: u32,
}

// SAFETY: `#[repr(C)]`, ten four-byte scalar fields with no padding (40
// bytes), matching the device-side struct byte for byte.
#[allow(unsafe_code)]
unsafe impl DeviceRepr for AdamArgs {}
const MAX_RANK: usize = 8;
const BLOCK: u32 = 256;
/// Full reductions at or above this size use the two-stage block kernel.
const FULL_REDUCE_THRESHOLD: usize = 32 * 1024;
const FULL_REDUCE_BLOCKS: usize = 64;

/// The strided-tensor descriptor the kernels take by value — the same
/// layout as `struct TensorMeta` in `kernels.cu` and `kernels.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct TensorMeta {
    rank: u32,
    offset: u32,
    dims: [u32; MAX_RANK],
    strides: [i32; MAX_RANK],
}

// SAFETY: `#[repr(C)]`, only `u32`/`i32` fields with no padding (2 + 8 + 8
// four-byte words = 72 bytes), matching the device-side struct byte for
// byte, so passing it as a by-value kernel parameter is well-defined.
#[allow(unsafe_code)]
unsafe impl DeviceRepr for TensorMeta {}

impl TensorMeta {
    fn from_layout(layout: &Layout, op: &'static str) -> Result<Self> {
        Self::from_parts(
            layout.shape.dims(),
            layout.strides.values(),
            layout.offset,
            op,
        )
    }

    fn from_parts(
        dims: &[usize],
        strides: &[isize],
        offset: usize,
        op: &'static str,
    ) -> Result<Self> {
        if dims.len() > MAX_RANK {
            return Err(Error::InvalidArgument {
                op,
                detail: format!(
                    "cuda kernels support rank <= {MAX_RANK}, got {}",
                    dims.len()
                ),
            });
        }
        let mut meta = TensorMeta {
            rank: dims.len() as u32,
            offset: offset as u32,
            dims: [1; MAX_RANK],
            strides: [0; MAX_RANK],
        };
        for (i, (&d, &s)) in dims.iter().zip(strides).enumerate() {
            meta.dims[i] = d as u32;
            meta.strides[i] = s as i32;
        }
        Ok(meta)
    }
}

/// The allocation a CUDA tensor's storage holds, behind `OpaqueBuffer`.
struct CudaBuf {
    slice: CudaSlice<f32>,
    device_index: usize,
}

/// A CUDA device as an oxmera backend.
pub struct CudaBackend {
    index: usize,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // Held so the functions' module outlives them.
    _module: Arc<CudaModule>,
    funcs: HashMap<&'static str, CudaFunction>,
}

/// Whether a CUDA driver library can be loaded on this machine. Cheap, and
/// the only question asked before deciding not to register a device: cudarc
/// itself panics on a missing library, so this is checked first.
pub fn is_driver_present() -> bool {
    cudarc::get_lib_name_candidates("cuda").iter().any(|name| {
        // SAFETY: dlopen of a shared library by name. libcuda's constructors
        // have no preconditions of ours to violate; a failed load returns
        // Err rather than aborting. Nothing from the library is called here.
        #[allow(unsafe_code)]
        let loaded = unsafe { libloading::Library::new(name) };
        loaded.is_ok()
    })
}

/// Register CUDA device 0 when a driver and a device exist. Quiet when the
/// machine has no CUDA at all; loud only when a device is present but the
/// kernels cannot be loaded, which is a bug worth seeing.
pub fn register_default() {
    // Idempotent, like the Metal backend's: a registered device keeps its
    // backend (and its context and stream).
    if oxmera_tensor::backend::backend_for(Device::Cuda { index: 0 }).is_ok() {
        return;
    }
    if !is_driver_present() {
        return;
    }
    match CudaContext::device_count() {
        Ok(n) if n > 0 => {}
        _ => return,
    }
    match CudaBackend::new(0) {
        Ok(backend) => register_backend(Arc::new(backend)),
        Err(e) => eprintln!("oxmera-cuda: disabled: {e}"),
    }
}

/// Name, total memory in bytes, and compute capability of CUDA device 0,
/// or `None` when there is no driver or no device.
pub fn device_summary() -> Option<(String, usize, (i32, i32))> {
    if !is_driver_present() {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    Some((
        ctx.name().ok()?,
        ctx.total_mem().ok()?,
        ctx.compute_capability().ok()?,
    ))
}

fn drv(op: &'static str) -> impl Fn(cudarc::driver::DriverError) -> Error {
    move |e| Error::Backend {
        op,
        detail: format!("{e:?}"),
    }
}

impl CudaBackend {
    /// Open device `index`, load the kernels, and keep one stream.
    pub fn new(index: usize) -> Result<Self> {
        let ctx = CudaContext::new(index).map_err(drv("cuda context"))?;
        let stream = ctx.default_stream();
        let module = match ctx.load_module(Ptx::from_src(KERNELS_PTX)) {
            Ok(m) => m,
            Err(first) => {
                // The driver did not accept the shipped PTX (typically an
                // older driver than the toolkit that emitted it): rebuild
                // from source for this device through NVRTC.
                let (major, minor) = ctx.compute_capability().map_err(drv("cuda cc"))?;
                let arch = format!("sm_{major}{minor}");
                let opts = cudarc::nvrtc::CompileOptions {
                    arch: Some(Box::leak(arch.into_boxed_str())),
                    ..Default::default()
                };
                let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNELS_CU, opts).map_err(|e| {
                    Error::Backend {
                        op: "cuda kernels",
                        detail: format!(
                            "driver rejected the shipped PTX ({first:?}) and NVRTC could not rebuild it: {e:?}"
                        ),
                    }
                })?;
                ctx.load_module(ptx).map_err(drv("cuda module"))?
            }
        };
        let mut funcs = HashMap::new();
        for &name in KERNEL_NAMES {
            funcs.insert(
                name,
                module.load_function(name).map_err(drv("cuda function"))?,
            );
        }
        Ok(Self {
            index,
            ctx,
            stream,
            _module: module,
            funcs,
        })
    }

    /// The device's name, as the driver reports it.
    pub fn device_name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| "CUDA device".into())
    }

    fn buf_of<'t>(&self, t: &'t Tensor, op: &'static str) -> Result<&'t CudaSlice<f32>> {
        if t.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: t.dtype(),
                op,
            });
        }
        let mismatch = || Error::DeviceMismatch {
            lhs: t.device(),
            rhs: self.device(),
            op,
        };
        let buf = t
            .storage()
            .opaque()
            .and_then(|o| o.inner().downcast_ref::<CudaBuf>())
            .ok_or_else(mismatch)?;
        if buf.device_index != self.index {
            return Err(mismatch());
        }
        Ok(&buf.slice)
    }

    /// A zeroed device buffer for `numel` elements. CUDA rejects a
    /// zero-byte allocation, so an empty tensor gets a one-element
    /// placeholder; the tensor's own length stays 0.
    fn alloc_out(&self, numel: usize) -> Result<CudaSlice<f32>> {
        self.stream
            .alloc_zeros::<f32>(numel.max(1))
            .map_err(drv("cuda alloc"))
    }

    fn wrap(&self, slice: CudaSlice<f32>, shape: Shape) -> Result<Tensor> {
        let numel = shape.numel();
        let buf = OpaqueBuffer::new(
            Arc::new(CudaBuf {
                slice,
                device_index: self.index,
            }),
            numel,
        );
        let storage = Storage::from_opaque(buf, DType::F32, self.device());
        Tensor::from_storage(Arc::new(storage), Layout::contiguous(shape))
    }

    /// Run a fully populated launch. Call sites build the argument list
    /// with `self.builder(name)` in the kernel's signature order.
    fn run(&self, mut launch: LaunchArgs<'_>, cfg: LaunchConfig, name: &'static str) -> Result<()> {
        // SAFETY: every call site pushes exactly the parameters the named
        // kernel declares in kernels.cu, in order and of matching type, and
        // every device pointer is a live allocation covering the extent the
        // kernel indexes (bounded by the numel/out_numel argument or the
        // m/k/n extents passed with it).
        #[allow(unsafe_code)]
        unsafe { launch.launch(cfg) }.map_err(drv(name))?;
        Ok(())
    }

    fn builder(&self, name: &'static str) -> LaunchArgs<'_> {
        self.stream.launch_builder(&self.funcs[name])
    }

    fn linear(numel: usize) -> LaunchConfig {
        let n = numel.max(1) as u32;
        LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn read_f32(&self, slice: &CudaSlice<f32>, len: usize) -> Result<Vec<f32>> {
        let mut host = vec![0.0f32; len];
        if len > 0 {
            let view = slice.slice(0..len);
            self.stream
                .memcpy_dtoh(&view, &mut host)
                .map_err(drv("cuda download"))?;
        }
        self.stream.synchronize().map_err(drv("cuda sync"))?;
        Ok(host)
    }

    /// Upload a validated `u32` index list for the gather/scatter kernels
    /// (a one-element placeholder when empty; CUDA rejects zero bytes).
    fn index_slice(&self, idx: &[u32]) -> Result<CudaSlice<u32>> {
        if idx.is_empty() {
            return self.stream.alloc_zeros::<u32>(1).map_err(drv("cuda alloc"));
        }
        self.stream
            .clone_htod(idx)
            .map_err(drv("cuda index upload"))
    }

    /// Strided-to-contiguous copy through the identity unary kernel.
    fn copy_strided(&self, a: &Tensor) -> Result<Tensor> {
        let input = self.buf_of(a, "contiguous")?;
        let numel = a.numel();
        let mut out = self.alloc_out(numel)?;
        if numel > 0 {
            let meta = TensorMeta::from_layout(a.layout(), "contiguous")?;
            let (op, n) = (0u32, numel as u32);
            let mut b = self.builder("unary_strided");
            b.arg(input).arg(&mut out).arg(&meta).arg(&op).arg(&n);
            self.run(b, Self::linear(numel), "unary_strided")?;
        }
        self.wrap(out, a.shape().clone())
    }
}

fn unary_opcode(op: UnaryOp) -> u32 {
    match op {
        UnaryOp::Neg => 1,
        UnaryOp::Exp => 2,
        UnaryOp::Ln => 3,
        UnaryOp::Abs => 4,
        UnaryOp::Sqrt => 5,
        UnaryOp::Sin => 6,
        UnaryOp::Cos => 7,
        UnaryOp::Tanh => 8,
        UnaryOp::Relu => 9,
        UnaryOp::Gelu => 10,
        UnaryOp::Sigmoid => 11,
        _ => u32::MAX,
    }
}

fn binary_opcode(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 1,
        BinaryOp::Sub => 2,
        BinaryOp::Mul => 3,
        BinaryOp::Div => 4,
        BinaryOp::Pow => 5,
        BinaryOp::Maximum => 6,
        BinaryOp::Minimum => 7,
        BinaryOp::Gt => 8,
        BinaryOp::Eq => 9,
        _ => u32::MAX,
    }
}

fn reduce_opcode(op: ReduceOp, ctx: &'static str) -> Result<u32> {
    match op {
        ReduceOp::Sum => Ok(0),
        ReduceOp::Max => Ok(1),
        ReduceOp::Min => Ok(2),
        _ => Err(Error::NotImplemented {
            op: ctx,
            detail: format!("cuda kernel for {op:?}"),
        }),
    }
}

/// Validate an `I64` index tensor against `extent` and pack it as `u32`.
/// Indices live on the host (this backend carries `f32` only).
fn index_list(indices: &Tensor, extent: usize, shape: &Shape) -> Result<Vec<u32>> {
    let idx = indices.to_device(Device::Cpu)?.to_vec_i64()?;
    let mut out = Vec::with_capacity(idx.len());
    for &i in &idx {
        if i < 0 || i as usize >= extent {
            return Err(Error::IndexOutOfBounds {
                index: vec![i.max(0) as usize],
                shape: shape.clone(),
            });
        }
        out.push(i as u32);
    }
    Ok(out)
}

fn split_axes(dims: &[usize], axes: &[usize], keepdim: bool) -> (Vec<usize>, Vec<usize>) {
    let mut out_dims = Vec::new();
    let mut kept = Vec::new();
    for (i, &d) in dims.iter().enumerate() {
        if axes.contains(&i) {
            if keepdim {
                out_dims.push(1);
            }
        } else {
            out_dims.push(d);
            kept.push(i);
        }
    }
    (out_dims, kept)
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        Device::Cuda { index: self.index }
    }

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn unary(&self, op: UnaryOp, a: &Tensor) -> Result<Tensor> {
        let input = self.buf_of(a, "unary")?;
        let numel = a.numel();
        let mut out = self.alloc_out(numel)?;
        if numel > 0 {
            let meta = TensorMeta::from_layout(a.layout(), "unary")?;
            let (opcode, n) = (unary_opcode(op), numel as u32);
            let mut b = self.builder("unary_strided");
            b.arg(input).arg(&mut out).arg(&meta).arg(&opcode).arg(&n);
            self.run(b, Self::linear(numel), "unary_strided")?;
        }
        self.wrap(out, a.shape().clone())
    }

    fn binary(&self, op: BinaryOp, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_shapes(a.shape(), b.shape())?;
        let av = a.broadcast_view(&out_shape)?;
        let bv = b.broadcast_view(&out_shape)?;
        let ab = self.buf_of(&av, "binary")?;
        let bb = self.buf_of(&bv, "binary")?;
        let numel = out_shape.numel();
        let mut out = self.alloc_out(numel)?;
        if numel > 0 {
            let ma = TensorMeta::from_layout(av.layout(), "binary")?;
            let mb = TensorMeta::from_layout(bv.layout(), "binary")?;
            let (opcode, n) = (binary_opcode(op), numel as u32);
            let mut l = self.builder("binary_strided");
            l.arg(ab)
                .arg(bb)
                .arg(&mut out)
                .arg(&ma)
                .arg(&mb)
                .arg(&opcode)
                .arg(&n);
            self.run(l, Self::linear(numel), "binary_strided")?;
        }
        self.wrap(out, out_shape)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = if a.layout().is_contiguous() && a.layout().offset == 0 {
            a.clone()
        } else {
            self.contiguous(a)?
        };
        let b = if b.layout().is_contiguous() && b.layout().offset == 0 {
            b.clone()
        } else {
            self.contiguous(b)?
        };
        let MatmulPlan {
            batch,
            m,
            k,
            n,
            a_batch_stride,
            b_batch_stride,
            out_shape,
        } = plan_matmul(a.shape(), b.shape())?;
        let ab = self.buf_of(&a, "matmul")?;
        let bb = self.buf_of(&b, "matmul")?;
        let mut out = self.alloc_out(batch * m * n)?;
        if batch * m * n > 0 {
            const TILE: u32 = 16;
            let cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(TILE), (m as u32).div_ceil(TILE), 1),
                block_dim: (TILE, TILE, 1),
                shared_mem_bytes: 0,
            };
            let (mu, ku, nu) = (m as u32, k as u32, n as u32);
            for bi in 0..batch {
                // A broadcast operand has batch stride 0: every batch reads
                // the same block, nothing is materialized.
                let a_base = (bi * a_batch_stride) as u32;
                let b_base = (bi * b_batch_stride) as u32;
                let c_base = (bi * m * n) as u32;
                let mut l = self.builder("matmul_tiled");
                l.arg(ab)
                    .arg(bb)
                    .arg(&mut out)
                    .arg(&mu)
                    .arg(&ku)
                    .arg(&nu)
                    .arg(&a_base)
                    .arg(&b_base)
                    .arg(&c_base);
                self.run(l, cfg, "matmul_tiled")?;
            }
        }
        self.wrap(out, out_shape)
    }

    fn reduce(&self, op: ReduceOp, a: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        let input = self.buf_of(a, "reduce")?;
        let dims = a.dims().to_vec();
        let strides = a.layout().strides.values().to_vec();
        let numel = a.numel();
        let opcode = reduce_opcode(op, "reduce")?;

        let full = axes.len() == dims.len();
        if full && numel >= FULL_REDUCE_THRESHOLD {
            let mut partials = self.alloc_out(FULL_REDUCE_BLOCKS)?;
            let meta = TensorMeta::from_layout(a.layout(), "reduce")?;
            let n = numel as u32;
            let cfg = LaunchConfig {
                grid_dim: (FULL_REDUCE_BLOCKS as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = self.builder("reduce_full_partials");
            l.arg(input)
                .arg(&mut partials)
                .arg(&meta)
                .arg(&opcode)
                .arg(&n);
            self.run(l, cfg, "reduce_full_partials")?;
            let host = self.read_f32(&partials, FULL_REDUCE_BLOCKS)?;
            let acc = host
                .into_iter()
                .fold(op.identity(), |acc, x| op.combine(acc, x));
            let out_shape = Shape::new(if keepdim { vec![1; dims.len()] } else { vec![] });
            return self.upload(&Tensor::from_vec_f32(vec![acc], out_shape)?);
        }

        let (out_dims, kept) = split_axes(&dims, axes, keepdim);
        let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();
        let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
        let red_dims: Vec<usize> = axes.iter().map(|&ax| dims[ax]).collect();
        let red_strides: Vec<isize> = axes.iter().map(|&ax| strides[ax]).collect();
        let kept_meta =
            TensorMeta::from_parts(&kept_dims, &kept_strides, a.layout().offset, "reduce")?;
        let red_meta = TensorMeta::from_parts(&red_dims, &red_strides, 0, "reduce")?;
        let out_shape = Shape::new(out_dims);
        let out_numel = out_shape.numel();
        let mut out = self.alloc_out(out_numel)?;
        if out_numel > 0 {
            let n = out_numel as u32;
            let mut l = self.builder("reduce_axis");
            l.arg(input)
                .arg(&mut out)
                .arg(&kept_meta)
                .arg(&red_meta)
                .arg(&opcode)
                .arg(&n);
            self.run(l, Self::linear(out_numel), "reduce_axis")?;
        }
        self.wrap(out, out_shape)
    }

    fn argmax(&self, a: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        let host = self.download(a)?;
        oxmera_tensor::backend::backend_for(Device::Cpu)?.argmax(&host, dim, keepdim)
    }

    fn contiguous(&self, a: &Tensor) -> Result<Tensor> {
        self.copy_strided(a)
    }

    fn download(&self, a: &Tensor) -> Result<Tensor> {
        let contiguous = if a.layout().is_contiguous() && a.layout().offset == 0 {
            a.clone()
        } else {
            self.copy_strided(a)?
        };
        let slice = self.buf_of(&contiguous, "to_cpu")?;
        let data = self.read_f32(slice, contiguous.numel())?;
        Tensor::from_vec_f32(data, a.shape().clone())
    }

    fn index_select(&self, a: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let dims = a.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidArgument {
                op: "index_select",
                detail: format!("dim {dim} out of range for rank {}", dims.len()),
            });
        }
        let input = self.buf_of(a, "index_select")?;
        let idx = index_list(indices, dims[dim], a.shape())?;
        let mut out_dims = dims.to_vec();
        out_dims[dim] = idx.len();
        let out_shape = Shape::new(out_dims);
        let out_numel = out_shape.numel();
        let mut out = self.alloc_out(out_numel)?;
        if out_numel > 0 {
            let idx_dev = self.index_slice(&idx)?;
            let meta = TensorMeta::from_layout(a.layout(), "index_select")?;
            let (d, l, n) = (dim as u32, idx.len() as u32, out_numel as u32);
            let mut b = self.builder("gather_dim");
            b.arg(input)
                .arg(&idx_dev)
                .arg(&mut out)
                .arg(&meta)
                .arg(&d)
                .arg(&l)
                .arg(&n);
            self.run(b, Self::linear(out_numel), "gather_dim")?;
        }
        self.wrap(out, out_shape)
    }

    fn index_add(&self, a: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        let dims = a.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidArgument {
                op: "index_add",
                detail: format!("dim {dim} out of range for rank {}", dims.len()),
            });
        }
        let idx = index_list(indices, dims[dim], a.shape())?;
        let mut expected = dims.to_vec();
        expected[dim] = idx.len();
        if src.dims() != expected.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: Shape::new(expected),
                got: src.shape().clone(),
                op: "index_add",
            });
        }
        let ab = self.buf_of(a, "index_add")?;
        let sb = self.buf_of(src, "index_add")?;
        let numel = a.numel();
        let mut out = self.alloc_out(numel)?;
        if numel > 0 {
            let idx_dev = self.index_slice(&idx)?;
            let ma = TensorMeta::from_layout(a.layout(), "index_add")?;
            let ms = TensorMeta::from_layout(src.layout(), "index_add")?;
            let (d, l, n) = (dim as u32, idx.len() as u32, numel as u32);
            let mut b = self.builder("scatter_add_dim");
            b.arg(ab)
                .arg(sb)
                .arg(&idx_dev)
                .arg(&mut out)
                .arg(&ma)
                .arg(&ms)
                .arg(&d)
                .arg(&l)
                .arg(&n);
            self.run(b, Self::linear(numel), "scatter_add_dim")?;
        }
        self.wrap(out, a.shape().clone())
    }

    fn adam_step(&self, step: &AdamStep<'_>) -> Result<(Tensor, Tensor, Tensor)> {
        let numel = step.param.numel();
        if step.grad.dims() != step.param.dims() {
            return Err(Error::ShapeMismatch {
                expected: step.param.shape().clone(),
                got: step.grad.shape().clone(),
                op: "adam_step",
            });
        }
        let dense = |t: &Tensor| -> Result<Tensor> {
            if t.layout().is_contiguous() && t.layout().offset == 0 {
                Ok(t.clone())
            } else {
                self.copy_strided(t)
            }
        };
        let p = dense(step.param)?;
        let g = dense(step.grad)?;
        let (m, v) = match (step.m, step.v) {
            (Some(m), Some(v)) => (dense(m)?, dense(v)?),
            _ => (p.clone(), p.clone()),
        };
        let pb = self.buf_of(&p, "adam_step")?;
        let gb = self.buf_of(&g, "adam_step")?;
        let mb = self.buf_of(&m, "adam_step")?;
        let vb = self.buf_of(&v, "adam_step")?;
        let mut p_out = self.alloc_out(numel)?;
        let mut m_out = self.alloc_out(numel)?;
        let mut v_out = self.alloc_out(numel)?;
        if numel > 0 {
            let args = AdamArgs {
                lr: step.lr,
                beta1: step.beta1,
                beta2: step.beta2,
                eps: step.eps,
                weight_decay: step.weight_decay,
                bc1: step.bias_correction1,
                bc2: step.bias_correction2,
                decoupled: u32::from(step.decoupled),
                has_state: u32::from(step.m.is_some() && step.v.is_some()),
                numel: numel as u32,
            };
            let mut b = self.builder("adam_step");
            b.arg(pb)
                .arg(gb)
                .arg(mb)
                .arg(vb)
                .arg(&mut p_out)
                .arg(&mut m_out)
                .arg(&mut v_out)
                .arg(&args);
            self.run(b, Self::linear(numel), "adam_step")?;
        }
        let shape = step.param.shape().clone();
        Ok((
            self.wrap(p_out, shape.clone())?,
            self.wrap(m_out, shape.clone())?,
            self.wrap(v_out, shape)?,
        ))
    }

    fn upload(&self, a: &Tensor) -> Result<Tensor> {
        let host = a.contiguous_untracked()?;
        let data = host.to_vec_f32()?;
        // Allocate first, copy only what exists: an empty tensor keeps its
        // one-element placeholder and copies nothing (the Metal backend's
        // issue #17 was reading past an empty Vec here).
        let mut slice = self.alloc_out(data.len())?;
        if !data.is_empty() {
            self.stream
                .memcpy_htod(&data, &mut slice)
                .map_err(drv("cuda upload"))?;
        }
        self.wrap(slice, a.shape().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcodes_match_the_kernel_tables() {
        // kernels.cu: unary 1..=11, binary 1..=9, reduce 0..=2. Any
        // renumbering must change both files together.
        assert_eq!(unary_opcode(UnaryOp::Neg), 1);
        assert_eq!(unary_opcode(UnaryOp::Sigmoid), 11);
        assert_eq!(binary_opcode(BinaryOp::Add), 1);
        assert_eq!(binary_opcode(BinaryOp::Eq), 9);
        assert_eq!(reduce_opcode(ReduceOp::Sum, "t").unwrap(), 0);
        assert_eq!(reduce_opcode(ReduceOp::Min, "t").unwrap(), 2);
    }

    #[test]
    fn tensor_meta_is_seventy_two_bytes_of_integers() {
        assert_eq!(std::mem::size_of::<TensorMeta>(), 72);
        let meta = TensorMeta::from_parts(&[2, 3], &[3, 1], 5, "t").unwrap();
        assert_eq!((meta.rank, meta.offset), (2, 5));
        assert_eq!(&meta.dims[..2], &[2, 3]);
        assert_eq!(&meta.strides[..2], &[3, 1]);
        assert!(TensorMeta::from_parts(&[1; 9], &[0; 9], 0, "t").is_err());
    }

    #[test]
    fn the_shipped_ptx_declares_every_kernel() {
        for name in KERNEL_NAMES {
            assert!(
                KERNELS_PTX.contains(&format!(".visible .entry {name}(")),
                "{name} missing from kernels.ptx — regenerate it from kernels.cu"
            );
            assert!(KERNELS_CU.contains(&format!("__global__ void {name}(")));
        }
        assert!(KERNELS_PTX.contains(".target sm_75"));
    }
}
