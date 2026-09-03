# ADR-0008 — Asynchronous dispatch, stateful backends, and the fused optimizer step

Status: Accepted
Date: 2026-09-03

## Context

Every Metal op encoded one command buffer and blocked on it
(`wait_until_completed`), and every CUDA op ran on one stream with a
synchronous download at the end. For the workloads that motivated 0.3.0 —
a downstream research pipeline training stacked small models on a few
thousand rows — the GPU spent its time on dispatch, not arithmetic: its
device-bench measured Metal at 19.4–19.6 s against 6.0–6.4 s on the CPU
for 100 epochs of three-seed fits (`oxmega device-bench`, Apple M4 Pro,
oxmera 0.2.0). Issue #28 asked for ops to be batched into one command
buffer and for the optimizer's dozen tiny elementwise ops to become one
launch.

## Decision

1. **Metal dispatch is asynchronous.** Ops are encoded into an open
   command buffer, one compute encoder each (so they execute in order,
   with Metal's hazard tracking between them), and the buffer is committed
   — without waiting — every 64 ops or at the first host read. `download`,
   the full reduction's host finish, and `to_vec` synchronize; nothing else
   does. The observable per-op semantics are unchanged: a tensor's bytes
   are final by the time any CPU code can see them.

2. **A host read waits on every outstanding command buffer, not the
   newest.** Command buffers on one queue *start* in commit order, but
   Metal may overlap them and finish them out of order. The first
   implementation waited on the newest committed buffer only and read
   zeros in three parity tests under `--test-threads`; the fix keeps every
   committed-but-unfinished buffer in a list, waits on all of them, and
   prunes those whose status is `Completed`. Entries stay recorded while
   a wait is in flight so a second synchronizing thread finds them too.

3. **Backend registration is idempotent.** `register_default()` on Metal
   and CUDA returns if the device already has a backend. A backend that
   holds an open command buffer cannot be replaced while tensors are in
   flight: the replaced instance's encoded work is never committed and the
   new instance has nothing to wait for, so a reader sees an unwritten
   output. The parity suite, which registers from every test, exposed
   exactly this. `register_backend` itself keeps its documented
   replace-semantics for callers who mean it.

4. **One fused Adam/AdamW kernel per parameter.** `Backend::adam_step`
   takes the parameter, its gradient, the optional moment state and the
   step's scalars and returns the three new tensors; Metal and CUDA
   implement it with the composite step's exact formula, and the optimizer
   uses it for parameters on a non-CPU device, falling back to the
   composite path when a backend declines. Parity tests hold the fused
   update to the composite CPU update within `f32` rounding over several
   steps, with and without decoupled decay.

5. **CUDA keeps its one stream.** cudarc's default stream already queues
   launches asynchronously; only `download` synchronizes. The A10G session
   for this release verified the new kernels (gather/scatter, `adam_step`)
   for parity and under Compute Sanitizer; no CUDA throughput number is
   claimed until it is measured on the benchmark that motivated the work.

## Consequences

- Measured on the motivating benchmark (`oxmega device-bench`, M4 Pro,
  back-to-back with the published 0.2.0, three rounds): Metal 19.4–19.6 s
  → 7.5–7.7 s with asynchronous dispatch alone (2.6×), → 7.4–7.6 s with
  the fused step as well; CPU 6.0–6.4 s in every configuration. The GPU is
  now within 1.2× of the CPU on a workload built to favour the CPU; it does
  not overtake it, and the docs say so.
- A backend is now a stateful object with a mutex. Encoders are still
  created, used and ended inside one call under that mutex — the
  thread-safety argument in `MetalBackend`'s `Send`/`Sync` impls is
  extended, not replaced.
- Reading device memory from Rust without going through `download`/
  `read_f32` is a bug by construction: only those paths synchronize. There
  are no such reads in the crate; the invariant is stated here so the next
  one is not written.
- `FLUSH_EVERY` bounds how much unsubmitted work one commit carries. It is
  a latency knob, not a correctness one; the tests pass at 1 and at 64.
