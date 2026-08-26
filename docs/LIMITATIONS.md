# Limitations

Read this before depending on oxmera. It is updated before every release;
numbers come from re-runnable commands.

## Scope

- **dtypes:** compute ops are `f32`; `i64` tensors exist for indices,
  argmax results, and class targets; `f64`/`u8` are storage-level only.
  No implicit promotion — dtype mismatches are typed errors.
- **matmul ranks:** rank-2 and batched rank-3. No einsum, no implicit
  batching of higher ranks.
- **CUDA: none.** The backend set is CPU and Apple Metal. The
  convergence-checked CUDA path (`cuda-oxide`/`reconverge`/`launchbound`)
  is deferred; its scaffolding lives in `research/` and those tools remain
  banned as dependencies of the stable workspace.

## Performance honesty

- The CPU backend is rayon-parallel with a cache-blocked GEMM, but it is
  not BLAS; nothing here competes with Accelerate/MKL and no such claim is
  made.
- The Metal backend dispatches one synchronous command buffer per op.
  For the demo's tiny batches, the CPU is *faster* (measured 2026-08-22:
  ~17k samples/s CPU vs ~5.7k Metal on a 32-sample-batch MLP, Apple M4
  Pro) — per-dispatch overhead dominates until tensors are large. Batching
  encoders and buffer pooling are roadmap.
- The CPU↔Metal parity tests bound elementwise disagreement at `1e-5`
  (relative), matmul at `1e-5·√k` — summation order differs between
  backends, and that is inherent to floating point, not a bug.

## Autograd

- Reverse mode only; no higher-order gradients (the tape is not itself
  differentiable).
- `requires_grad_(true)` on a non-leaf does not retain intermediate
  gradients (leaves only).
- Gradients at non-differentiable points follow the usual conventions
  (`relu'(0) = 0`; `max` ties split evenly); gradcheck avoids these
  points deliberately.

## Backends and dispatch

- Gather/scatter (`index_select`/`index_add`) execute on the CPU; Metal
  tensors round-trip through host memory for those ops (and for `argmax`,
  whose result is always a CPU `i64` tensor).
- Backend registration is process-global. Metal registers at load time on
  macOS; if a linker strips the registration (unusual link setups), call
  `oxmera::init()` explicitly.
- `Dropout` masks are generated on the CPU per forward pass.

## Terminal surfaces

- Golden tests run the dashboard in **replay mode** from fixtures; the
  live `--tui` path shares the renderer but its clock-derived numbers
  (throughput) are asserted only in replay.
- `BatchNorm2d` running statistics and `Dropout` mode are process-local
  state (`set_training`), not serialized by safetensors.
