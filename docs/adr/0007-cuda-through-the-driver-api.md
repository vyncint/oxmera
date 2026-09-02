# ADR-0007 — CUDA through the driver API, beside the research line

Status: Accepted
Date: 2026-09-02

## Context

oxmera had two CUDA-shaped things and no CUDA backend. `Device::Cuda`
existed as a handle that every path rejected as "deferred". Under
`research/`, a crate of kernels written in Rust with `cuda-oxide` was
statically verified by `reconverge` and `launchbound` on plain CI runners
and, in one metered session, executed correctly on an A10G. The research
line's value is that verification; turning it into the shipped backend
would have meant taking `cuda-oxide` — a git-pinned nightly-only compiler
backend behind the dependency firewall — into the stable workspace, or
maintaining a host runtime around it, and shipping fixed-size research
kernels that are not a general backend.

## Decision

The shipped backend is `crates/oxmera-cuda`, standard CUDA:

- **Kernels in CUDA C** (`kernels.cu`), a one-to-one port of the Metal MSL
  kernels — same `TensorMeta` ABI, same opcodes — so the two GPU backends
  share one host-side contract and the same parity suite shape.
- **PTX shipped in the crate**, compiled once by `nvcc` for the
  `compute_75` virtual architecture (Turing and newer, JIT-forward). The
  CUDA C source is embedded as well: a driver that rejects the shipped PTX
  ISA gets the kernels rebuilt for its compute capability through NVRTC.
  No build-time toolkit, no `build.rs` invoking `nvcc`.
- **The driver API through `cudarc`** with dynamic loading: `libcuda` is
  `dlopen`ed at runtime. A machine without it has no CUDA device — exactly
  Metal's behaviour off macOS — and the crate's own tests pass there.
  `cudarc` panics when the library is missing, so the backend probes for
  it first and never reaches that path.
- **The hub stays independent of `cudarc`.** `oxmera-tensor` gains
  `StorageData::Opaque` (an `Arc<dyn Any>` plus an element count); the
  backend downcasts its own allocation type. Metal keeps its dedicated
  variant for now.
- **The research line stays research**, renamed `oxmera-cuda-oxide` so
  the two are named for what they are. Its gate (`reconverge` strict at
  both compute capabilities, `launchbound prune` fully admitted) is
  unchanged and still runs on every push. The dependency firewall is
  unchanged: `cuda-oxide`, `reconverge` and `launchbound` remain banned
  from the stable workspace; `cudarc` is not in those families.

## Consequences

- `Device::Cuda` is real: `to_device`, every op, autograd and
  `oxmera train --device cuda` work, verified at `1e-5` against the CPU
  backend on an NVIDIA A10G and clean under Compute Sanitizer.
- Correctness-first: one stream, synchronous downloads, no cuBLAS, `f32`
  only, `argmax`/`index_*` via a CPU round-trip like Metal. Throughput
  work is roadmap and will be measured, not estimated.
- Regenerating the PTX is a documented manual step
  (`nvcc -arch=compute_75 -O3 -ptx kernels.cu -o kernels.ptx`); a unit
  test asserts the checked-in PTX declares every kernel the source does.
- Adding `cudarc` and `libloading` to the stable graph passed `cargo deny`
  (licenses MIT/Apache-2.0 and ISC, no banned names, crates.io only).
- Two backends porting one kernel set exposed a device-generic autograd
  bug: `backward()` seeded gradients on the CPU regardless of the loss's
  device. It is fixed in the hub and the end-to-end test now exists for
  both Metal and CUDA.
