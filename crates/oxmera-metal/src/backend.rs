//! The Metal backend implementation (macOS only).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use metal::{
    Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device as MtlDevice,
    MTLResourceOptions, MTLSize,
};
use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{DType, Device, Error, Layout, Result, Shape};
use oxmera_tensor::backend::{
    AdamStep, Backend, BinaryOp, MatmulPlan, ReduceOp, UnaryOp, plan_matmul, register_backend,
};
use oxmera_tensor::storage::{MetalBuffer, Storage, StorageData};
use oxmera_tensor::tensor::Tensor;

const KERNELS: &str = include_str!("../kernels.metal");
const MAX_RANK: usize = 8;
const FULL_REDUCE_THRESHOLD: usize = 32 * 1024;
/// Dispatches encoded into one command buffer before it is committed
/// (without waiting). Bounds the work a single commit carries; the queue
/// executes committed buffers in order, so correctness never depends on
/// this number.
const FLUSH_EVERY: usize = 64;

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
/// pipelines for every kernel (elementwise, reductions, matmul,
/// gather/scatter).
///
/// Dispatch is asynchronous: ops are encoded into an open command buffer
/// (one compute encoder each, so they execute in order with Metal's
/// hazard tracking between them) and the buffer is committed every
/// `FLUSH_EVERY` (64) ops or at the first host read. Nothing waits until a
/// tensor's bytes are actually needed on the CPU — `download`, the full
/// reduction's partials, `to_vec` — which is what turns a training step of
/// a hundred tiny ops from a hundred round trips into one.
pub struct MetalBackend {
    index: usize,
    device: MtlDevice,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    pending: Mutex<Pending>,
}

/// The open command buffer and every one committed but not yet known to
/// have completed. Command buffers on one queue *start* in commit order,
/// but Metal may overlap them and finish them out of order, so waiting on
/// the newest is not enough — measured: three parity tests read zeros
/// under `--test-threads` with a newest-only wait.
#[derive(Default)]
struct Pending {
    open: Option<CommandBuffer>,
    encoded: usize,
    committed: Vec<CommandBuffer>,
}

impl Pending {
    fn commit_open(&mut self) {
        if let Some(open) = self.open.take() {
            open.commit();
            self.committed.push(open);
        }
        self.encoded = 0;
    }

    fn prune_completed(&mut self) {
        self.committed
            .retain(|cb| cb.status() != metal::MTLCommandBufferStatus::Completed);
    }
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("index", &self.index)
            .field("device", &self.device.name().to_string())
            .finish()
    }
}

// SAFETY: MTLDevice, MTLCommandQueue, MTLCommandBuffer and
// MTLComputePipelineState are documented thread-safe by Apple; command
// encoders (the one Metal type that is not) are created, used, and ended
// inside a single call while the `pending` mutex is held, and never
// stored. The pipeline map is immutable after construction; the open
// command buffer is only ever touched under that mutex.
#[allow(unsafe_code)]
unsafe impl Send for MetalBackend {}
// SAFETY: see the `Send` justification above.
#[allow(unsafe_code)]
unsafe impl Sync for MetalBackend {}

/// Register a backend for the system-default Metal device, when one
/// exists. Idempotent: a device that already has a backend keeps it. The
/// backend carries state (the open command buffer of asynchronous
/// dispatch), so replacing it while tensors are in flight would leave
/// their encoded work uncommitted — the parity suite, which calls this
/// from every test, read zeros under `--test-threads` before this check.
pub fn register_default() {
    if oxmera_tensor::backend::backend_for(Device::Metal { index: 0 }).is_ok() {
        return;
    }
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
    "gather_dim",
    "scatter_add_dim",
    "adam_step",
];

/// Host mirror of `struct AdamArgs` in `kernels.metal`/`kernels.cu`.
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

impl AdamArgs {
    fn new(step: &AdamStep<'_>, numel: usize) -> Self {
        Self {
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
        }
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: #[repr(C)], ten four-byte fields, no padding (40 bytes).
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(
                (self as *const AdamArgs).cast::<u8>(),
                std::mem::size_of::<AdamArgs>(),
            )
        }
    }
}

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
            pending: Mutex::new(Pending::default()),
        })
    }

    /// Commit whatever is encoded and block until every committed command
    /// buffer has completed. Called before any host read of device memory;
    /// harmless when nothing is pending.
    pub fn synchronize(&self) {
        // Commit under the lock, wait outside it, on every outstanding
        // buffer. Entries stay recorded while a wait is in flight so a
        // second thread synchronizing concurrently finds and waits on them
        // too; completed buffers are pruned afterwards (waiting on a
        // completed buffer returns at once, so a stale entry is harmless).
        let outstanding: Vec<CommandBuffer> = {
            let mut pending = self.pending.lock().expect("metal pending poisoned");
            pending.commit_open();
            pending.committed.clone()
        };
        for cb in &outstanding {
            cb.wait_until_completed();
        }
        self.pending
            .lock()
            .expect("metal pending poisoned")
            .prune_completed();
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
            _ => Err(Error::DeviceMismatch {
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

    /// Encode one compute pass into the open command buffer. Returns as
    /// soon as it is encoded; execution is ordered after every earlier
    /// pass and completes by the next [`MetalBackend::synchronize`].
    fn run_sync(
        &self,
        pipeline: &'static str,
        buffers: &[&Buffer],
        small_args: &[&[u8]],
        grid: MTLSize,
        group: MTLSize,
    ) {
        let mut pending = self.pending.lock().expect("metal pending poisoned");
        let cb = pending
            .open
            .get_or_insert_with(|| self.queue.new_command_buffer().to_owned());
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
        pending.encoded += 1;
        if pending.encoded >= FLUSH_EVERY {
            pending.commit_open();
            pending.prune_completed();
        }
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

    /// Read a buffer's f32 contents (unified memory), first completing
    /// every encoded and committed pass so the bytes are final.
    fn read_f32(&self, buffer: &Buffer, len: usize) -> Vec<f32> {
        self.synchronize();
        // SAFETY: `contents()` on a StorageModeShared buffer is a valid
        // pointer to `length()` bytes of unified memory; every kernel that
        // wrote it was awaited (`synchronize`) before this read, and `len`
        // is bounded by the allocation the caller made.
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

/// Validate an `I64` index tensor against `extent` and pack it as the
/// `u32` list the gather/scatter kernels read. Indices live on the host
/// (the GPU backends carry `f32` only), so this is where they are checked.
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

impl MetalBackend {
    fn index_buffer(&self, idx: &[u32]) -> Buffer {
        if idx.is_empty() {
            return self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
        }
        self.device.new_buffer_with_data(
            idx.as_ptr().cast(),
            (idx.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
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
        let MatmulPlan {
            batch,
            m,
            k,
            n,
            a_batch_stride,
            b_batch_stride,
            out_shape,
        } = plan_matmul(a.shape(), b.shape())?;
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
            // A broadcast operand has batch stride 0: every batch reads
            // the same block, nothing is materialized.
            let a_base = (bi * a_batch_stride) as u32;
            let b_base = (bi * b_batch_stride) as u32;
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

    fn index_select(&self, a: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let dims = a.dims();
        if dim >= dims.len() {
            return Err(Error::InvalidArgument {
                op: "index_select",
                detail: format!("dim {dim} out of range for rank {}", dims.len()),
            });
        }
        let input = self.buffer_of(a, "index_select")?;
        let idx = index_list(indices, dims[dim], a.shape())?;
        let mut out_dims = dims.to_vec();
        out_dims[dim] = idx.len();
        let out_shape = Shape::new(out_dims);
        let out_numel = out_shape.numel();
        let out = self.alloc_out(out_numel);
        if out_numel > 0 {
            let idx_buf = self.index_buffer(&idx);
            let meta = TensorMeta::from_layout(a.layout(), "index_select")?;
            let (d, l, n) = (dim as u32, idx.len() as u32, out_numel as u32);
            let (grid, group) = self.linear_grid(out_numel);
            self.run_sync(
                "gather_dim",
                &[input, &idx_buf, &out],
                &[
                    meta.as_bytes(),
                    &d.to_ne_bytes(),
                    &l.to_ne_bytes(),
                    &n.to_ne_bytes(),
                ],
                grid,
                group,
            );
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
        let ab = self.buffer_of(a, "index_add")?;
        let sb = self.buffer_of(src, "index_add")?;
        let numel = a.numel();
        let out = self.alloc_out(numel);
        if numel > 0 {
            let idx_buf = self.index_buffer(&idx);
            let ma = TensorMeta::from_layout(a.layout(), "index_add")?;
            let ms = TensorMeta::from_layout(src.layout(), "index_add")?;
            let (d, l, n) = (dim as u32, idx.len() as u32, numel as u32);
            let (grid, group) = self.linear_grid(numel);
            self.run_sync(
                "scatter_add_dim",
                &[ab, sb, &idx_buf, &out],
                &[
                    ma.as_bytes(),
                    ms.as_bytes(),
                    &d.to_ne_bytes(),
                    &l.to_ne_bytes(),
                    &n.to_ne_bytes(),
                ],
                grid,
                group,
            );
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
        // Contiguous, offset-free inputs: the kernel indexes linearly.
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
            // First step: the kernel ignores these; hand it the parameter
            // buffer so every binding is a real allocation.
            _ => (p.clone(), p.clone()),
        };
        let pb = self.buffer_of(&p, "adam_step")?;
        let gb = self.buffer_of(&g, "adam_step")?;
        let mb = self.buffer_of(&m, "adam_step")?;
        let vb = self.buffer_of(&v, "adam_step")?;
        let (p_out, m_out, v_out) = (
            self.alloc_out(numel),
            self.alloc_out(numel),
            self.alloc_out(numel),
        );
        if numel > 0 {
            let args = AdamArgs::new(step, numel);
            let (grid, group) = self.linear_grid(numel);
            self.run_sync(
                "adam_step",
                &[pb, gb, mb, vb, &p_out, &m_out, &v_out],
                &[args.as_bytes()],
                grid,
                group,
            );
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
        // Metal returns nil for a zero-length buffer, so an empty tensor
        // still gets a one-element placeholder — but it must be
        // *allocated*, never *copied*: `new_buffer_with_data` reads the
        // requested byte count from the host pointer, and an empty Vec's
        // pointer owns zero bytes (issue #17 was a SIGSEGV here).
        let buffer = if data.is_empty() {
            self.alloc_out(0)
        } else {
            self.device.new_buffer_with_data(
                data.as_ptr().cast(),
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
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
