# Architecture

Design decisions live in [docs/adr/](docs/adr/); ADR-0004 (dynamic backend
dispatch) and ADR-0006 (the pivot to full implementation) shape most of
what follows.

## Layers

```
oxmera            umbrella re-export; links every backend for the platform
oxmera-core       dtype, shape, strides, layout, device handles, errors
oxmera-tensor     THE HUB: Tensor, views, storage (CPU + Metal buffers),
                  the Backend trait + registry, autograd tape, operator
                  overloads, and the built-in CPU backend implementation
oxmera-ops        the backend op vocabulary (re-exported from the hub)
oxmera-runtime    init/registration, device selection, no_grad re-exports
oxmera-cpu        public face of the CPU backend
oxmera-metal      Apple Metal backend: MSL kernels, pipelines, buffers
oxmera-autograd   autograd surface + finite-difference gradcheck
oxmera-nn         Module, layers, losses, initializers, safetensors
oxmera-optim      SGD / Adam / AdamW / RMSprop over shared Param handles
oxmera-cli        `oxmera doctor` and `oxmera train --tui`
--- separate nightly workspace: research/ ---
oxmera-cuda       deferred CUDA path (cuda-oxide; nothing implemented)
```

## Why the tensor crate is the hub

Rust's orphan rule requires `std::ops` impls for `Tensor` to live in the
crate that defines `Tensor`; dispatching those operators requires the
`Backend` trait and registry to be visible there too. And the CPU
reference backend lives *inside* the hub so `backend_for(Device::Cpu)` can
register it lazily — the CPU is always available, with no link-order or
life-before-main caveats. GPU backends register from load-time
constructors when linked (`oxmera-metal` on macOS), with
`oxmera::init()` as the explicit fallback.

## Dispatch and autograd

- `Device` is a handle; the registry resolves it to an `Arc<dyn Backend>`.
- Backends implement a small primitive set: `unary`/`binary` (strided,
  broadcasting), `matmul` (rank-2 and batched rank-3), `reduce`,
  `argmax`, `contiguous`, `upload`/`download`, and optional
  `index_select`/`index_add` (with a CPU round-trip fallback).
- Everything else — mean, softmax, losses, convolution (im2col),
  normalization — is composed from primitives device-generically, so the
  autograd tape differentiates composites for free.
- The tape lives on the tensor: ops record a `GradFn` (inputs + VJP
  closure) when recording is on and an input is tracked; `backward()`
  walks reverse topological order, accumulating into leaf gradients.
  `no_grad` is a thread-local RAII switch.

## The Metal backend

MSL kernels compiled at runtime into pipelines: strided elementwise
(broadcast via stride-0 dims), per-output-element axis reductions, a
threadgroup-memory two-stage full reduction, and a 16×16 tiled GEMM.
Buffers are `StorageModeShared` (unified memory); every dispatch is a
synchronous command buffer in v0.1. Parity with the CPU backend is
asserted at `1e-5` in tests that run on real hardware.

## The dependency firewall (unchanged)

No stable-workspace crate may depend — directly or transitively — on the
`cuda-oxide`, `reconverge`, or `launchbound` families. Those serve the
deferred CUDA path as *tools*; `deny.toml` enforces the ban and CI fails
on violations. The `research/` workspace (own pinned nightly,
`nightly-2026-04-03`) is where that path lives until it returns.

## Two workspaces, still

The root workspace is stable Rust (MSRV 1.88, measured against the
lockfile). `research/` keeps its own `rust-toolchain.toml`; a fresh clone
builds and tests with no CUDA toolkit and no LLVM, and CI's Linux runners
prove the no-GPU path on every push (Metal code is cfg-gated to macOS).
