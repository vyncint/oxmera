# Limitations

Read this before depending on oxmera. It is updated before every release;
numbers come from re-runnable commands.

## Scope

- **`nn` and `optim` are `f32`, whatever the input dtype.** Every layer's
  parameters are `f32`; there is no dtype argument on any constructor and
  no `Module::to_dtype`. Passing an `f64` tensor to a layer is a typed
  error naming the layer and the cast to make. The intended `f64` workflow
  is tensor-level maths *outside* the model — research metrics, set
  likelihoods, normalisers — which is what it was added for. Making `nn`
  hold `f64` parameters is deferred on purpose, not missing by accident:
  it would drag the optimizer state with it and nobody has asked.
- **Reductions over a zero-length extent** do not agree with each other,
  and the disagreement is deliberate:

  | reduction | over an empty extent | why |
  | --- | --- | --- |
  | `sum` | `0` | the identity, and it composes under further reduction |
  | `max` / `min` | `-inf` / `+inf` | the identities, and they compose |
  | `mean` | **typed error** | `0/0`; a `NaN` here composes into every gradient and surfaces an epoch later |
  | `argmax` | **typed error** | there is no index to return |

  `mean` returned `NaN` before 0.4.0. Reducing a *non-empty* axis of a
  tensor that merely has an empty one is unaffected — `[0,3].mean(&[1])`
  is `[0]`, and divides by 3.
- **dtypes:** `f32` is the compute dtype on every device. `f64` tensors
  are CPU-only: storage, every CPU primitive, autograd and `to_dtype`
  (`F32` ↔ `F64`, `I64` → float) work; moving one to a GPU is a typed
  error, and so is mixing dtypes in one op — there is no implicit
  promotion. `i64` tensors exist for indices, argmax results and class
  targets; `u8` is storage-level only. The f64 path is the plain reference
  formula, parallel but untuned: correct first, not fast.
- **matmul ranks:** any rank ≥ 2 with NumPy batch broadcasting; ranks above
  3 are lowered onto the rank-3 backend contract (a broadcast batch is
  materialized when it must be repeated). `einsum` covers one- and
  two-operand contractions with an explicit output; diagonals (`ii->i`)
  and implicit output are typed errors.
- **linear algebra:** small batched `cholesky`/`logdet`/`det` (SPD only,
  non-PD is a typed error naming the batch) and `eigh` (symmetric, Jacobi,
  not differentiable). They run on the CPU in `f64` internally; a GPU
  tensor round-trips through the host for them. No general LU, solve, or
  SVD.

  `eigh` **requires** a symmetric input and says so: a matrix whose
  `|a[i][j] - a[j][i]|` exceeds `EIGH_SYMMETRY_TOL` (`1e-5`, relative to
  the matrix's own largest magnitude) is a typed error naming the batch
  index and the worst pair. The tolerance is the same bound the CPU↔GPU
  parity suite uses, so a covariance or Gram matrix assembled as `XᵀX / n`
  in `f32` — symmetric in intent, asymmetric in the last few bits — still
  passes. Before 0.4.0 the full matrix was read and silently symmetrized,
  so a genuinely non-symmetric input returned the eigenpairs of
  `(A + Aᵀ)/2` with no error.
- **The CUDA backend is a default feature, and can be turned off.**
  `cargo add oxmera --no-default-features` drops `oxmera-cuda` — and with
  it `cudarc`, `libloading`, the `ctor`/`dtor` pair, 88 KB of embedded PTX,
  and a pre-`main` constructor that `dlopen`s `libcuda`. Seven crates leave
  the graph, 47 → 40. `Device::Cuda` remains a *typed runtime error* in
  that configuration, never a compile error, so code naming the device
  still builds. Metal stays gated by target rather than by feature,
  because it cannot build off macOS at all. Before 0.4.0 CUDA was
  mandatory everywhere, including on macOS.
- **CUDA: correctness-first.** `Device::Cuda` runs the same eight kernels
  as Metal through the CUDA driver API on one stream; launches queue
  asynchronously and `download` synchronizes. There is no cuBLAS path, no
  multi-stream overlap, no `f16`, and no performance claim beyond the
  parity and sanitizer evidence in the changelog — no CUDA throughput has
  been measured on the benchmark that motivated 0.3.0. Requirements: an NVIDIA driver from the 12.8 series or newer
  (the `cuda-12080` bindings); the shipped PTX targets `compute_75`
  (Turing+) and was emitted by CUDA 13.2, so a driver older than that
  needs `libnvrtc` present for the runtime rebuild. The research line under
  `research/oxmera-cuda-oxide` (`cuda-oxide`/`reconverge`/`launchbound`)
  is unrelated to this backend and still not a dependency.
## Performance honesty

- The CPU backend is rayon-parallel with a cache-blocked GEMM, but it is
  not BLAS; nothing here competes with Accelerate/MKL and no such claim is
  made.
- The Metal backend encodes ops asynchronously into one command buffer and
  waits only at host reads, and the optimizer step is one fused launch per
  parameter (0.3.0). On the small-model research benchmark that motivated
  the change the GPU is still not faster than the CPU: 7.4–7.6 s Metal vs
  6.1–6.4 s CPU for 100 epochs of three stacked seeds (Apple M4 Pro,
  measured 2026-09-03; 19.4–20.1 s Metal before). Per-dispatch cost is
  amortized, per-op cost is not — tensors have to be large for a GPU to
  win, and there is still no buffer pooling and no MPS/cuBLAS matmul.
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

- `argmax` and the linear-algebra factorizations (`cholesky`, `eigh`)
  round-trip GPU tensors through host memory; `argmax`'s result is always
  a CPU `i64` tensor. Index tensors are CPU `i64` tensors on every device
  (the GPU kernels read a packed `u32` copy). `index_add` on the GPU
  scans the index list per output element — deterministic and exact, but
  O(index count) per element; it is meant for the narrow VJP, `cat`/`pad`
  and embedding-sized index lists, not for scattering millions of rows.
- Backend registration is process-global and idempotent: a device that has
  a backend keeps it (a backend now carries dispatch state). Metal and
  CUDA register at load time; if a linker strips the registration (unusual
  link setups), call `oxmera::init()` explicitly.
- `Dropout` masks are generated on the CPU per forward pass.

## Terminal surfaces

- Golden tests run the dashboard in **replay mode** from fixtures; the
  live `--tui` path shares the renderer but its clock-derived numbers
  (throughput) are asserted only in replay.
- `BatchNorm2d` running statistics and `Dropout` mode are process-local
  state (`set_training`), not serialized by safetensors.
