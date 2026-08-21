# ADR-0002 — Local execution backend on Apple Silicon

Status: Proposed
Date: 2026-08-21

## Context

Tier-C exercises and the `oxmera-metal` crate need a way to execute on the
GPU in the development machine. Constraints: it must be feature-gated and
macOS-only, it must not shape the CUDA backend's traits around its own
abstractions, and there is **no convergence gate on this path** — that
banner follows whatever is chosen.

## Options

### 1. `objc2-metal` (direct Metal bindings)

- (+) Teaches actual Metal: command queues, pipelines, threadgroup memory —
  the closest analogue to what the CUDA path teaches, which is the point of
  a learning project.
- (+) No abstraction layer between the exercise and the hardware; kernel
  sources are plain MSL strings compiled at runtime.
- (−) The maintainer writes all the plumbing (device, queue, encoder
  lifecycle) — more room, but genuinely more work.
- (−) macOS-only skills; nothing transfers to other platforms.

### 2. `wgpu`

- (+) Portable (Vulkan/DX12/Metal); one shader language (WGSL); the
  best-documented Rust GPU stack.
- (−) Teaches wgpu's abstraction, not the GPU: barriers, workgroup memory,
  and dispatch semantics arrive pre-abstracted — the learning the project
  exists for happens inside the library instead of in the exercises.
- (−) WGSL kernel skills do not transfer to the cuda-oxide path (Rust
  `#[kernel]`s) at all.

### 3. CubeCL

- (+) One kernel language (Rust) for both CUDA and Metal targets; closest
  to "write once".
- (−) A large framework dependency whose abstractions would dictate the
  `Backend` trait shape — exactly what this ADR must avoid; and it
  overlaps cuda-oxide's role, muddying what tier B teaches.

### 4. CPU-only for now (defer the choice)

- (+) Zero cost, zero constraint; tier C simply waits.
- (−) Tier C is the only free *execution* tier; deferring it removes the
  one place a real GPU can be touched for $0.

## Decision (proposed)

Option 1, `objc2-metal`, chosen for teaching value: it keeps the Metal path
honest (a real queue, a real pipeline, a real threadgroup) and constrains
the `Backend` trait not at all, because the trait already speaks only in
tensors. Not scaffolded yet — `oxmera-metal` is created when tier C opens,
so switching to option 2 or 4 costs nothing today.
