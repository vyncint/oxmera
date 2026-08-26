//! The Metal backend implementation (macOS only).

use std::collections::HashMap;
use std::sync::Arc;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device as MtlDevice,
    MTLResourceOptions, MTLSize,
};
use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{DType, Device, Error, Layout, Result, Shape};
use oxmera_tensor::backend::{Backend, BinaryOp, ReduceOp, UnaryOp, register_backend};
use oxmera_tensor::storage::{MetalBuffer, Storage, StorageData};
use oxmera_tensor::tensor::Tensor;

const KERNELS: &str = include_str!("../kernels.metal");
const MAX_RANK: usize = 8;
const FULL_REDUCE_THRESHOLD: usize = 32 * 1024;

/// The strided-tensor descriptor shared with the MSL kernels. Layout must
/// match `TensorMeta` in `kernels.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct TensorMeta {
    rank: u32,
    offset: u32,
    dims: [u32; MAX_RANK],
    strides: [i32; MAX_RANK],
}

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
                    "metal kernels support rank <= {MAX_RANK}, got {}",
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

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: TensorMeta is #[repr(C)], contains only integers (no
        // padding at the tail: 2×u32 + 8×u32 + 8×i32 = 72 bytes, all
        // 4-aligned), so viewing it as bytes is well-defined.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(
                (self as *const TensorMeta).cast::<u8>(),
                std::mem::size_of::<TensorMeta>(),
            )
        }
    }
}

/// The Metal backend: one device, one command queue, precompiled compute
/// pipelines for every kernel.
pub struct MetalBackend {
    index: usize,
    device: MtlDevice,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("index", &self.index)
            .field("device", &self.device.name().to_string())
            .finish()
    }
}

// SAFETY: MTLDevice, MTLCommandQueue, and MTLComputePipelineState are
// documented thread-safe by Apple; command encoders (the one Metal type
// that is not) are created, used, and ended inside a single call and never
// stored. The pipeline map is immutable after construction.
#[allow(unsafe_code)]
unsafe impl Send for MetalBackend {}
// SAFETY: see the `Send` justification above.
#[allow(unsafe_code)]
unsafe impl Sync for MetalBackend {}

/// Register a backend for the system-default Metal device, when one
/// exists. Safe to call repeatedly.
pub fn register_default() {
    if let Some(device) = MtlDevice::system_default() {
        match MetalBackend::new(device, 0) {
            Ok(backend) => register_backend(Arc::new(backend)),
            Err(e) => {
                // A machine with a broken Metal stack still gets a working
                // CPU; surfacing this at registration would abort main.
                eprintln!("oxmera-metal: disabled: {e}");
            }
        }
    }
}

/// Description of the Metal device for `oxmera doctor`.
pub fn device_summary() -> Option<(String, u64, u64)> {
    let device = MtlDevice::system_default()?;
    Some((
        device.name().to_string(),
        device.recommended_max_working_set_size(),
        device.current_allocated_size(),
    ))
}

const PIPELINE_NAMES: &[&str] = &[
    "unary_strided",
    "binary_strided",
    "reduce_axis",
    "reduce_full_partials",
    "matmul_tiled",
];

impl MetalBackend {
    /// Compile the kernel library and build every pipeline on `device`.
    pub fn new(device: MtlDevice, index: usize) -> Result<Self> {
        let library = device
            .new_library_with_source(KERNELS, &CompileOptions::new())
            .map_err(|e| Error::Backend {
                op: "metal compile",
                detail: e.to_string(),
            })?;
        let mut pipelines = HashMap::new();
        for &name in PIPELINE_NAMES {
            let function = library
                .get_function(name, None)
                .map_err(|e| Error::Backend {
                    op: "metal function",
                    detail: e.to_string(),
                })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| Error::Backend {
                    op: "metal pipeline",
                    detail: e.to_string(),
                })?;
            pipelines.insert(name, pipeline);
        }
        let queue = device.new_command_queue();
        Ok(Self {
            index,
            device,
            queue,
            pipelines,
        })
    }

    fn buffer_of<'t>(&self, t: &'t Tensor, op: &'static str) -> Result<&'t Buffer> {
        if t.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: t.dtype(),
                op,
            });
        }
        match t.storage().data() {
            StorageData::Metal(b) if b.device_index == self.index => Ok(b.buffer()),
            StorageData::Metal(_) | StorageData::Cpu(_) => Err(Error::DeviceMismatch {
                lhs: t.device(),
                rhs: self.device(),
                op,
            }),
        }
    }

    fn alloc_out(&self, numel: usize) -> Buffer {
        let bytes = (numel.max(1) * 4) as u64;
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    fn wrap(&self, buffer: Buffer, shape: Shape) -> Result<Tensor> {
        let storage = Storage::from_metal(MetalBuffer::new(buffer, self.index), DType::F32);
        Tensor::from_storage(Arc::new(storage), Layout::contiguous(shape))
    }

    /// Encode one compute pass and block until it completes.
    fn run_sync(
        &self,
        pipeline: &'static str,
        buffers: &[&Buffer],
        small_args: &[&[u8]],
        grid: MTLSize,
        group: MTLSize,
    ) {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines[pipeline]);
        for (i, b) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (j, bytes) in small_args.iter().enumerate() {
            enc.set_bytes(
                (buffers.len() + j) as u64,
                bytes.len() as u64,
                bytes.as_ptr().cast(),
            );
        }
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    fn linear_grid(&self, numel: usize) -> (MTLSize, MTLSize) {
        let width = 256.min(numel.max(1)) as u64;
        (
            MTLSize {
                width: numel.max(1) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        )
    }

    /// Read a buffer's f32 contents (unified memory) after GPU work has
    /// completed.
    fn read_f32(&self, buffer: &Buffer, len: usize) -> Vec<f32> {
        // SAFETY: `contents()` on a StorageModeShared buffer is a valid
        // pointer to `length()` bytes of unified memory; every kernel that
        // wrote it was awaited (`wait_until_completed`) before this read,
        // and `len` is bounded by the allocation the caller made.
        #[allow(unsafe_code)]
        let slice = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) };
        slice.to_vec()
    }
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

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal { index: self.index }
    }

    fn name(&self) -> &'static str {
        "metal"
    }

    fn unary(&self, op: UnaryOp, a: &Tensor) -> Result<Tensor> {
        let input = self.buffer_of(a, "unary")?;
        let numel = a.numel();
        let out = self.alloc_out(numel);
        let meta = TensorMeta::from_layout(a.layout(), "unary")?;
        let opcode: u32 = unary_opcode(op);
        let n = numel as u32;
        let (grid, group) = self.linear_grid(numel);
        self.run_sync(
            "unary_strided",
            &[input, &out],
            &[meta.as_bytes(), &opcode.to_ne_bytes(), &n.to_ne_bytes()],
            grid,
            group,
        );
        self.wrap(out, a.shape().clone())
    }

    fn binary(&self, op: BinaryOp, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_shapes(a.shape(), b.shape())?;
        let av = a.broadcast_view(&out_shape)?;
        let bv = b.broadcast_view(&out_shape)?;
        let ab = self.buffer_of(&av, "binary")?;
        let bb = self.buffer_of(&bv, "binary")?;
        let numel = out_shape.numel();
        let out = self.alloc_out(numel);
        let ma = TensorMeta::from_layout(av.layout(), "binary")?;
        let mb = TensorMeta::from_layout(bv.layout(), "binary")?;
        let opcode: u32 = binary_opcode(op);
        let n = numel as u32;
        let (grid, group) = self.linear_grid(numel);
        self.run_sync(
            "binary_strided",
            &[ab, bb, &out],
            &[
                ma.as_bytes(),
                mb.as_bytes(),
                &opcode.to_ne_bytes(),
                &n.to_ne_bytes(),
            ],
            grid,
            group,
        );
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
        let (batch, m, k, n) = match (a.ndim(), b.ndim()) {
            (2, 2) => {
                let (m, ka) = (a.dims()[0], a.dims()[1]);
                let (kb, n) = (b.dims()[0], b.dims()[1]);
                if ka != kb {
                    return Err(Error::ShapeMismatch {
                        expected: Shape::from([ka, n]),
                        got: b.shape().clone(),
                        op: "matmul",
                    });
                }
                (1usize, m, ka, n)
            }
            (3, 3) => {
                let (ba, m, ka) = (a.dims()[0], a.dims()[1], a.dims()[2]);
                let (bb, kb, n) = (b.dims()[0], b.dims()[1], b.dims()[2]);
                if ba != bb || ka != kb {
                    return Err(Error::ShapeMismatch {
                        expected: Shape::from([ba, ka, n]),
                        got: b.shape().clone(),
                        op: "matmul",
                    });
                }
                (ba, m, ka, n)
            }
            (ra, rb) => {
                return Err(Error::InvalidArgument {
                    op: "matmul",
                    detail: format!("supported ranks are 2x2 and 3x3 (batched); got {ra}x{rb}"),
                });
            }
        };
        let ab = self.buffer_of(&a, "matmul")?;
        let bb = self.buffer_of(&b, "matmul")?;
        let out = self.alloc_out(batch * m * n);
        let tile = 16u64;
        let grid = MTLSize {
            width: (n as u64).div_ceil(tile) * tile,
            height: (m as u64).div_ceil(tile) * tile,
            depth: 1,
        };
        let group = MTLSize {
            width: tile,
            height: tile,
            depth: 1,
        };
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        // One synchronous pass per batch element keeps the kernel simple;
        // batched sizes in this project are small.
        for bi in 0..batch {
            let a_base = (bi * m * k) as u32;
            let b_base = (bi * k * n) as u32;
            let c_base = (bi * m * n) as u32;
            self.run_sync(
                "matmul_tiled",
                &[ab, bb, &out],
                &[
                    &mu.to_ne_bytes(),
                    &ku.to_ne_bytes(),
                    &nu.to_ne_bytes(),
                    &a_base.to_ne_bytes(),
                    &b_base.to_ne_bytes(),
                    &c_base.to_ne_bytes(),
                ],
                grid,
                group,
            );
        }
        let out_shape = if batch == 1 && a.ndim() == 2 {
            Shape::from([m, n])
        } else {
            Shape::from([batch, m, n])
        };
        self.wrap(out, out_shape)
    }

    fn reduce(&self, op: ReduceOp, a: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        let input = self.buffer_of(a, "reduce")?;
        let dims = a.dims().to_vec();
        let strides = a.layout().strides.values().to_vec();
        let numel = a.numel();
        let opcode: u32 = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Max => 1,
            ReduceOp::Min => 2,
            _ => {
                return Err(Error::NotImplemented {
                    op: "reduce",
                    detail: format!("metal kernel for {op:?}"),
                });
            }
        };

        let full = axes.len() == dims.len();
        if full && numel >= FULL_REDUCE_THRESHOLD {
            // Two-stage: threadgroup partials on the GPU, tiny finish on
            // the host.
            let groups = 64usize;
            let group_size = 256usize;
            let partials = self.alloc_out(groups);
            let meta = TensorMeta::from_layout(a.layout(), "reduce")?;
            let n = numel as u32;
            let grid = MTLSize {
                width: (groups * group_size) as u64,
                height: 1,
                depth: 1,
            };
            let group = MTLSize {
                width: group_size as u64,
                height: 1,
                depth: 1,
            };
            self.run_sync(
                "reduce_full_partials",
                &[input, &partials],
                &[meta.as_bytes(), &opcode.to_ne_bytes(), &n.to_ne_bytes()],
                grid,
                group,
            );
            let host = self.read_f32(&partials, groups);
            let acc = host
                .into_iter()
                .fold(op.identity(), |a, x| op.combine(a, x));
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
        let out = self.alloc_out(out_numel);
        let n = out_numel as u32;
        let (grid, group) = self.linear_grid(out_numel);
        self.run_sync(
            "reduce_axis",
            &[input, &out],
            &[
                kept_meta.as_bytes(),
                red_meta.as_bytes(),
                &opcode.to_ne_bytes(),
                &n.to_ne_bytes(),
            ],
            grid,
            group,
        );
        self.wrap(out, out_shape)
    }

    fn argmax(&self, a: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        // Indices are I64; kernels here are f32-only, so argmax runs on the
        // host over the downloaded values. The result is a CPU tensor,
        // which is where index math is consumed.
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
        let buffer = self.buffer_of(&contiguous, "to_cpu")?;
        let data = self.read_f32(buffer, contiguous.numel());
        Tensor::from_vec_f32(data, a.shape().clone())
    }

    fn upload(&self, a: &Tensor) -> Result<Tensor> {
        let host = a.contiguous_untracked()?;
        let data = host.to_vec_f32()?;
        let bytes = (data.len().max(1) * 4) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        self.wrap(buffer, a.shape().clone())
    }
}

impl MetalBackend {
    /// Strided-to-contiguous copy through the identity unary kernel.
    fn copy_strided(&self, a: &Tensor) -> Result<Tensor> {
        let input = self.buffer_of(a, "contiguous")?;
        let numel = a.numel();
        let out = self.alloc_out(numel);
        let meta = TensorMeta::from_layout(a.layout(), "contiguous")?;
        let opcode: u32 = 0; // identity
        let n = numel as u32;
        let (grid, group) = self.linear_grid(numel);
        self.run_sync(
            "unary_strided",
            &[input, &out],
            &[meta.as_bytes(), &opcode.to_ne_bytes(), &n.to_ne_bytes()],
            grid,
            group,
        );
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
