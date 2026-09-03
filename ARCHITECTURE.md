# Architecture

Design decisions live in [docs/adr/](docs/adr/); ADR-0004 (dynamic backend
dispatch) and ADR-0006 (the pivot to full implementation) shape most of
what follows.

## Layers

```
oxmera            umbrella re-export; links every backend for the platform
oxmera-core       dtype, shape, strides, layout, device handles, errors
oxmera-tensor     THE HUB: Tensor, views, storage (CPU, Metal, and opaque
                  device buffers), the Backend trait + registry, autograd
                  tape, operator overloads, and the built-in CPU backend
oxmera-ops        the backend op vocabulary (re-exported from the hub)
oxmera-runtime    init/registration, device selection, no_grad re-exports
oxmera-cpu        public face of the CPU backend
oxmera-metal      Apple Metal backend: MSL kernels, pipelines, buffers
oxmera-cuda       NVIDIA CUDA backend: CUDA C kernels as PTX, driver API
oxmera-autograd   autograd surface + finite-difference gradcheck
oxmera-nn         Module, layers, losses, initializers, safetensors
oxmera-optim      SGD / Adam / AdamW / RMSprop over shared Param handles
oxmera-cli        `oxmera doctor` and `oxmera train --tui`
--- separate nightly workspace: research/ ---
oxmera-cuda-oxide research CUDA kernels in Rust via cuda-oxide, verified
                  by reconverge + launchbound; never a dependency
```

## Why the tensor crate is the hub

Rust's orphan rule requires `std::ops` impls for `Tensor` to live in the
crate that defines `Tensor`; dispatching those operators requires the
`Backend` trait and registry to be visible there too. And the CPU
reference backend lives *inside* the hub so `backend_for(Device::Cpu)` can
register it lazily — the CPU is always available, with no link-order or
life-before-main caveats. GPU backends register from load-time
constructors when linked (`oxmera-metal` on macOS; `oxmera-cuda` wherever
`libcuda` can be loaded and a device exists), with `oxmera::init()` as the
explicit fallback. A backend the hub does not depend on keeps its
allocation behind `StorageData::Opaque` — an `Arc<dyn Any>` plus an
element count — and downcasts it back; that is how `oxmera-cuda` holds a
`cudarc` slice without `oxmera-tensor` knowing about `cudarc`.

## Dispatch and autograd

- `Device` is a handle; the registry resolves it to an `Arc<dyn Backend>`.
- Backends implement a small primitive set: `unary`/`binary` (strided,
  broadcasting), `matmul` (rank 2 or 3 with NumPy batch broadcasting, per
  the shared `plan_matmul` contract; ranks above 3 are lowered onto it in
  the method layer), `reduce`, `argmax`, `contiguous`, `upload`/`download`,
  and the optional `index_select`/`index_add`, `cholesky`/`eigh` and
  `adam_step` — each with a documented fallback (a CPU round-trip of every
  operand, or the composite optimizer step) when a backend declines.
- Everything else — mean, softmax, losses, convolution (im2col),
  normalization, `einsum`, `eye`/`diag`/`trace`, `logdet` — is composed
  from primitives device-generically, so the autograd tape differentiates
  composites for free. `cholesky` carries its own VJP (Murray 2016).
- `f32` is the compute dtype on every device; `f64` tensors exist on the
  CPU (storage, every CPU primitive, autograd) and convert with
  `to_dtype`. The GPU backends reject them with a typed error.
- The tape lives on the tensor: ops record a `GradFn` (inputs + VJP
  closure) when recording is on and an input is tracked; `backward()`
  walks reverse topological order, accumulating into leaf gradients.
  `no_grad` is a thread-local RAII switch.

## The Metal backend

MSL kernels compiled at runtime into pipelines: strided elementwise
(broadcast via stride-0 dims), per-output-element axis reductions, a
threadgroup-memory two-stage full reduction, and a 16×16 tiled GEMM.
Buffers are `StorageModeShared` (unified memory). Dispatch is
asynchronous since 0.3.0: ops are encoded into an open command buffer (one
encoder each, ordered with hazard tracking), committed every 64 ops or at
the first host read, which waits on every outstanding buffer — Metal may
finish command buffers out of commit order (ADR-0008). Gather/scatter and
the fused Adam step are kernels too, so nothing but `argmax` and the
linear-algebra factorizations round-trips through the host. Parity with
the CPU backend is asserted at `1e-5` in tests that run on real hardware,
and a concurrency stress hammers the asynchronous path from eight
threads.

## The CUDA backend

The same eight kernels as Metal, written once in CUDA C (`kernels.cu`) with
the identical `TensorMeta` ABI and opcodes, compiled by `nvcc` to PTX for
the `compute_75` virtual architecture and embedded in the crate; the CUDA
C source is embedded too, and a driver that rejects the shipped PTX ISA
gets the kernels rebuilt for its compute capability through NVRTC. The
driver library is `dlopen`ed by `cudarc` — nothing links against CUDA, so
the crate builds and its tests pass on machines with no toolkit and no
GPU, where it registers no device. One context, one stream (launches
queue asynchronously; only `download` synchronizes); parity with the CPU
backend is asserted at `1e-5` on real hardware, and the parity binary runs
clean under Compute Sanitizer.

## The dependency firewall (unchanged)

No stable-workspace crate may depend — directly or transitively — on the
`cuda-oxide`, `reconverge`, or `launchbound` families. Those are the
*research* CUDA toolchain, kept in the `research/` workspace (own pinned
nightly, `nightly-2026-04-03`) and verified on plain CI runners;
`deny.toml` enforces the ban and CI fails on violations. The shipped
`oxmera-cuda` backend uses `cudarc` and the driver API and has nothing in
common with that toolchain but the GPU.

## Two workspaces, still

The root workspace is stable Rust (MSRV 1.88, measured against the
lockfile). `research/` keeps its own `rust-toolchain.toml`; a fresh clone
builds and tests with no CUDA toolkit and no LLVM, and CI's Linux runners
prove the no-GPU path on every push (Metal code is cfg-gated to macOS;
CUDA simply finds no driver and registers nothing).
