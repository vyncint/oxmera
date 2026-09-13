# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **termlens 0.10.1 → 0.11**, with the vendored skill and the report
  action's `cli-version:` pins in `ci.yml` and `stress.yml`. 0.11
  is termlens's stability candidate: from it no promised item changes
  incompatibly before its 1.0, so this requirement should hold for a while.

  Its one breaking change lands here as a simplification.
  `Screen::unsupported()` returns a view instead of a slice of `Arc<str>`,
  and `unsupported_overflow()` folds into it — so the dashboard's pinned
  list and "the record is not truncated" are one assertion in
  `tests/emulation.rs`, and `doctor`'s "the report is plain text" is one
  call rather than two, because the view is non-empty when shapes
  overflowed as well as when any were retained. The `Vec<String>` helper
  that existed to make the comparison typecheck is gone.

  The pin's known-defect caveat goes with it: termlens#320, which named
  blink and strikethrough as unsupported although the attribute shadow
  implements them, was fixed upstream in 0.10.2.

  The bump moved three of the places that name the version and left two:
  `PIN_TERMLENS` in `pins.yml` and CONTRIBUTING's own line, both still at
  0.10.1. Nothing would have caught that until the weekly pins watch
  compared the stale pin with crates.io and filed a drift issue for a bump
  that had already happened. Both are corrected, and
  `check-skill-version.sh` now checks every entry on CONTRIBUTING's
  must-touch list instead of the vendored skill alone —
  `PIN_TERMLENS` against `Cargo.lock` exactly, because an exact version is
  what the watch compares, and the rest on major.minor as before.

- **termlens 0.9.0 → 0.10.1**, with the `serde` feature, and the terminal
  suite grown into the surface it opens. No call site broke: the crate never
  used `drag`, a `termlens::Style` literal, `assert_screen_snapshot!` or a
  `match` on `termlens::Error`, so the version bump is the manifest, the
  lockfile and the strings that name it. What the release buys is coverage
  that did not exist:

  - `tests/emulation.rs` — the invariant the rest of the suite rests on.
    `Screen::unsupported()` is pinned exactly: `["^[[59m"]` for the
    dashboard (ratatui's underline-colour reset, which changes no cell) and
    empty for `doctor`, whose report is plain text. A sequence joining
    either list means the grid the goldens were blessed from may be wrong.
    Plus `insert_mode`, `visual_bells`, `bells`, `links`, `title`,
    `mouse_modes` and both round trips — the snapshot text format and the
    JSON `TERMLENS_ARTIFACT_DIR` writes.
  - `tests/styles.rs` — the dashboard's colour, which the goldens
    structurally cannot see: they compare `to_string()`. Until now the loss
    and accuracy sparkline colours could be swapped, the header's bold
    dropped, or the gauge cyan lost, with every test still green.
  - `tests/report.rs` — the report at 80 columns measured rather than
    modelled (`row_wrapped`, with a 60-column leg so the claim is not
    vacuous), the rows that scroll off a 24-row terminal (`scrollback_*`,
    `locate`), and the first coverage `src/probe.rs` has ever had: the
    real-machine path, machine-dependent rows masked with `mask_rect` so the
    shape is assertable on any runner.
  - `tests/termlens_cli.rs` — `termlens inspect` driving the real binary and
    reproducing the committed goldens byte for byte, and `termlens
    diff`/`render` reading those goldens, so they stay the interchange
    format a bug report needs. `#[ignore]`d: a published crate must not
    `cargo install` behind a contributor's back; `stress.yml` runs them.
  - Both 100-iteration stresses now compare with `Screen::diff` instead of
    two normalized strings — stronger (it sees styles and the cursor) and
    readable when it fails (only the rows that moved, with a marker under
    each changed column).
  - CI writes `TERMLENS_ARTIFACT_DIR` and renders what a failure left behind
    into the job summary; a new `skill-version` job fails when the vendored
    agent skill drifts from the dependency.

## [0.4.0] — 2026-09-06

The audit milestone (#33–#40): what the framework claimed, checked against
what it did. Every item was found by exercising `main` as a downstream
consumer — building the workspace, driving the CLI through a pty, and
probing the public tensor surface op by op — and every one carries a
reproduction that was run rather than inferred from reading the source.

The design bar was that each fix leaves behind something that fails next
time. A refreshed capability list decays again by 0.5.0; a test that fails
when a release adds a capability and does not list it does not.

### Fixed

- **`eigh` silently symmetrized a non-symmetric input** (#33). It read the
  full matrix, averaged the triangles, and returned the eigenpairs of
  `(A + Aᵀ)/2` — a different matrix — with no error: `[[1, 2], [5, 1]]`
  answered `[-2.5, 4.5]` where the true eigenvalues are `1 ± √10`, and the
  returned pair did not satisfy `A v = λ v` for the `A` that was passed.
  Now a typed error naming the batch index, the worst offending pair and
  the cast to make, above a relative tolerance (`EIGH_SYMMETRY_TOL`,
  `1e-5`, now public) chosen to keep passing the case the check exists to
  permit: a covariance or Gram matrix assembled in `f32`, symmetric in
  intent and asymmetric in the last few bits. Every other precondition in
  the library was already a typed error; this was the one that was not.
- **Nothing checked that `kernels.ptx` was built from `kernels.cu`** (#34).
  The crate ships the kernels twice and `load_module` chooses between them
  by driver age, so a mismatch means two users on the same published
  version run different code — the newer driver takes the stale PTX, the
  older one rebuilds the edited source through NVRTC. The only guard
  compared kernel *names*: measured, changing `relu` from `fmaxf(x, 0.0f)`
  to `fmaxf(x, 1.0f)` passed every suite that runs without a GPU, because
  the eleven tests that would catch it are `#[ignore]`d without hardware.
  Now `kernels.ptx.source` carries a fingerprint of the source, written by
  the same `just ptx` recipe that writes the PTX, and asserted by a test
  that needs no GPU. The same mutation now fails it, with the command to
  fix it in the message.
- **`train --tui` did not give the terminal back when signalled** (#40).
  Measured through a pty: after `SIGTERM` the shell was left inside the
  alternate screen with the cursor hidden, and nothing written to say so —
  the way out is to type `reset` blind. `ratatui::init()` already installed
  a panic hook, so this was the one remaining unguarded exit. `SIGINT`,
  `SIGTERM` and `SIGHUP` now restore and **re-raise with the default
  disposition**, so the process still reports as killed by that signal to
  `$?`, to `timeout` and to a supervisor. `signal-hook` was already in the
  tree through crossterm, so this is an import rather than a dependency.
- **`oxmera doctor` reported a capability list frozen before 0.2.0** (#36).
  Six string literals in the CLI, pinned by three golden frames, so adding
  a feature to the list cost more than leaving it stale — and CUDA, `f64`,
  the linear algebra, `einsum`, gather/scatter and per-group optimizer
  hyper-parameters were all absent, three releases on. The rows now live in
  the crates that implement them (`oxmera_tensor::CAPABILITIES` and
  siblings), so adding an op family is one line in the crate that adds it.
- **The crates.io descriptions for `oxmera` and `oxmera-tensor` still said
  "CPU and Apple Metal"** (#37), two releases after the CUDA backend
  shipped. That line is what appears in search results and `cargo search`.

### Changed

- **`mean` over a zero-length extent is a typed error** (#38), where it
  returned `NaN`. Four reductions over an empty extent behaved four ways;
  `sum` and `max` keep their identities because those compose correctly
  under further reduction, and `NaN` does not — it flows into the loss,
  then into every gradient, and surfaces an epoch later as a model that
  stopped learning for no visible reason. `argmax` already refused the same
  input for the same reason. Reducing a *non-empty* axis of a tensor that
  merely has an empty one is unaffected. The whole family is tabulated in
  `docs/LIMITATIONS.md`.
- **Passing a mismatched dtype to an `nn` layer names the layer** (#39),
  instead of reporting `dtype mismatch: expected F64, got F32 in matmul` —
  an internal operation the caller did not write. `nn` and `optim` are
  `f32` regardless of input dtype; that boundary is now stated in
  `docs/LIMITATIONS.md`, and `Module::to_dtype` is recorded as deliberately
  deferred rather than missing.
- **Pins: reconverge 0.4.0 → 0.5.0, launchbound 2.0.0 → 2.1.0.** The other
  two of the four pins did not move — reconverge 0.5.0 still records
  `nightly-2026-04-03` and cuda-oxide `a766fc26` — so this is the
  one-variable case, and the gate was re-measured rather than assumed:
  0 findings at cc 7.5 and 8.6, 12 admitted and 0 refused at both, and the
  over-cap tile still refused with RC004 at deny confidence. Identical to
  the 0.4.0/2.0.0 result. Recorded in `docs/research-baseline.md` §(e).
- **termlens 0.8.0 → 0.9.0**, and the pty tests moved to its idioms:
  `bin!` for the spawn, `snapshot_after` in place of
  `wait_until` + `wait_idle` + `screen()` (one settled instant instead of
  three, and it waits on the picture holding still rather than on bytes
  stopping), `wait_stable` after a resize, and `termlens::Result<()>` so a
  failure prints the grid the application was showing.

### Added

- **A `cuda` feature, on by default** (#35). `--no-default-features` drops
  `oxmera-cuda` and with it `cudarc`, `libloading` and the `ctor`/`dtor`
  pair — seven crates, 47 → 40 — plus 88 KB of embedded PTX and a
  pre-`main` constructor that `dlopen`s `libcuda`. All of that was
  previously mandatory on every platform, *including macOS*, where NVIDIA
  hardware cannot exist and Metal was correctly gated by target.
  `Device::Cuda` remains a typed runtime error with the feature off, never
  a compile error.
- **A second CI configuration.** Until now every job built the workspace
  defaults, so `--no-default-features` had never been compiled — which is
  how a mandatory CUDA backend went unnoticed for two releases. A feature
  nothing builds is a feature that is already broken. The job also asserts
  that `cudarc`, `libloading`, `ctor` and `oxmera-cuda` are absent from the
  graph.
- **`just ptx`**, which regenerates `kernels.ptx` and its fingerprint
  together. The command previously lived only in the maintainer's notes.
- **Guards, so each fix survives the next release.** A test that the facade
  description names every backend; one that `doctor` names every op family
  the release ships (whole-word — the first version used `contains` and
  passed with `eigh` deleted, because `weights` contains `eigh`); one that
  capability rows fit an 80-column terminal; FNV-1a's published test
  vectors, after the first draft of the fingerprint wrote the prime with
  twelve hex digits instead of eleven.

## [0.3.0] — 2026-09-03

The "what oxmega needs" milestone (#25–#30) plus the pins triage (#23):
everything a downstream research pipeline asked for after running on
0.2.0, each item measured or parity-tested before it was written down.

### Added

- **Native gather/scatter on Metal and CUDA** (#25): `index_select` and
  `index_add` run on the device through `gather_dim`/`scatter_add_dim`
  (one thread per output element scanning the index list — deterministic,
  adds in index order like the CPU reference, no atomics). The fallback
  path now moves *every* operand through the CPU round-trip; moving only
  the target left `src` on the device and failed the `narrow` VJP with a
  DeviceMismatch on CUDA (found by oxmega's k-DPP loss).
- **Small batched linear algebra** (#26): `Tensor::eye`/`eye_on`, `diag`,
  `diag_embed`, `trace`, `cholesky` (differentiable — Murray 2016's VJP,
  computed on the host in `f64`), `logdet`, `det` (SPD, via Cholesky) and
  `eigh` (symmetric, cyclic Jacobi, eigenvalues ascending, not
  differentiable). `cholesky`/`eigh` are `Backend` primitives with a CPU
  implementation and the same round-trip fallback as the index ops; a
  non-PD matrix is a typed error naming the batch index and pivot.
- **`f64` tensors on the CPU** (#27): `DType::F64` storage,
  `from_vec_f64`/`to_vec_f64`/`get_f64`, `Tensor::to_dtype` (`F32` ↔
  `F64`, `I64` → float; differentiable), and every CPU primitive in f64 —
  unary, broadcasting binary, matmul, Neumaier-compensated reductions,
  argmax, index ops, `cholesky`, `eigh`. Autograd follows the dtype. The
  GPU backends stay `f32`: `to_device` of an f64 tensor is a typed error.
- **Asynchronous Metal dispatch** (#28): ops are encoded into an open
  command buffer and committed every 64 ops or at the first host read,
  which waits on every outstanding buffer (ADR-0008). `register_default()`
  is idempotent on Metal and CUDA — a stateful backend must not be
  replaced mid-flight. A new eight-thread concurrency stress covers the
  asynchronous path.
- **Fused Adam/AdamW step** (#28): `Backend::adam_step` and an `adam_step`
  kernel on both GPU backends — one launch per parameter, the composite
  step kept as reference and fallback, parity-tested over several steps
  with and without decoupled decay.
- **Parameter groups** (#29): `ParamGroup { params, lr, weight_decay }`;
  `Sgd`/`Adam`/`AdamW`/`RmsProp::with_groups(..)` and `groups_mut()` for
  schedules. The plain constructors are the one-group case and produce
  identical updates (asserted).
- **Rank-4+ matmul broadcasting and `einsum`** (#30): every leading
  dimension is a batch dimension with NumPy broadcasting; ranks above 3
  are lowered onto the unchanged rank-3 backend contract from recorded
  view ops, so no new VJP. `einsum(spec, &[a, b])` covers the common one-
  and two-operand contractions (`ij,jk->ik`, `bij,bjk->bik`,
  `rbhd,rdo->rbho`, `ij->ji`, `i,i->`, `i,j->ij`); diagonals and implicit
  output are typed errors. Loop-checked and gradchecked.

### Changed

- **termlens 0.6.1 → 0.8.0** (#23). PTY suites and both 100-iteration
  stresses unchanged.
- **The pins watch compares cuda-oxide with what the pinned reconverge
  verifies, not upstream HEAD** (#23). reconverge is a rustc driver built
  on the nightly cuda-oxide needs; upstream HEAD moved to a newer nightly
  than reconverge 0.4.0 is built on, so the old watch reported drift
  nobody could act on. The cuda-oxide pin stays at `a766fc26`.
- `kernels.ptx` regenerated with nvcc 13.2 (`compute_75`, eight entries).

### Measured

- **Metal, oxmega device-bench** (Apple M4 Pro, 100 epochs × 3 stacked
  seeds, six fits, back-to-back with the published 0.2.0, three rounds):
  total 19.4–20.1 s → 7.5–7.7 s with asynchronous dispatch alone (2.6×)
  → 7.4–7.6 s with the fused optimizer step; CPU 6.0–6.4 s in every
  configuration. Per fit: mlp-h16 BCE 1.50 → 0.33 s, linear-lag5 set-NLL
  3.43 → 0.87 s, deepsets-H30 set-NLL 6.04 → 2.92 s. The GPU is within
  1.2× of the CPU on a workload built to favour the CPU; it does not
  overtake it.
- **CUDA on an NVIDIA A10G** (sm_86, driver 595.71.05, CUDA 13.2, rustc
  1.98.0): parity suite 11/11 incl. index ops with duplicate indices and
  strided views, `narrow` backward on the device, and the fused Adam step;
  full workspace tests green with CUDA registered at load time;
  Compute Sanitizer memcheck 0 errors and racecheck 0 hazards over the
  whole parity suite single-threaded (one parallel run reported 1
  racecheck hazard that five reruns did not reproduce), synccheck 0
  errors; `oxmera train --device cuda --epochs 3` reaches loss 0.6693 at
  ~47k samples/s — the same trajectory as 0.2.0. No CUDA throughput
  claim beyond that.

### Fixed

- Parity tests under `--test-threads` read zeros from Metal once dispatch
  was asynchronous: two causes, both fixed — waiting on the newest
  committed command buffer only (Metal finishes them out of order) and
  re-registering a fresh backend from every test (uncommitted work left
  behind in the replaced instance).

## [0.2.0] — 2026-09-02

The NVIDIA CUDA release, plus the edge-case hardening milestone.

### Added

- **`oxmera-cuda`: a real `Device::Cuda` backend** through the CUDA driver
  API via `cudarc`. The Metal kernel set (strided unary/binary, axis and
  two-stage full reductions, 16×16 tiled matmul) ported to CUDA C, compiled
  once to PTX for `compute_75` and embedded; the source is embedded too and
  is rebuilt through NVRTC for drivers that reject the shipped ISA. `libcuda`
  is loaded at runtime — building needs no toolkit, and a machine without a
  driver simply has no CUDA device. Registered by `oxmera::init()` and at
  load time; `default_device()` prefers Metal, then CUDA, then CPU;
  `oxmera train --device cuda`; `oxmera doctor` reports the device, memory
  and compute capability. (ADR-0007.)
- `StorageData::Opaque` — device buffers owned by a backend crate the hub
  does not depend on.
- `Tensor::is_tracked()` — the public predicate for autograd-tape
  membership, which is what observes `no_grad` (#20).
- `matmul` batch broadcasting and rank-2 × rank-3 operands, per the shared
  `plan_matmul` contract every backend implements; broadcast operands use a
  zero batch stride, nothing is materialized; VJPs reduce onto the operand
  shape (#22).
- `oxmera --help` / `-h` / `help` print usage to stdout and exit 0 (#21).

### Measured

- **CUDA on an NVIDIA A10G** (sm_86, driver 595.71.05, CUDA 13.2): parity
  suite 8/8 at `1e-5` (unary, binary with broadcast, reductions on every
  axis and in full, matmul incl. batch broadcast, views/softmax/argmax,
  zero-element tensors, end-to-end autograd); full workspace 108 tests
  green with CUDA registered at load time; Compute Sanitizer memcheck 0
  errors, racecheck 0 hazards, synccheck 0 errors over the parity binary;
  `oxmera train --device cuda --epochs 10` reaches the same loss as
  CPU/Metal (0.643) at ~14.7k samples/s. Correctness only — no throughput
  claims.
- **CPU performance** (Apple M4 Pro, release, best of 3, 0.1.1 → 0.2.0):
  full `sum` of 10M elements 6.6 → 193 GB/s; `mean` over [4096,2048] 4.7 →
  180 GB/s; axis-1 sum 38 → 167 GB/s; axis-0 sum 8.8 → 90 GB/s; same-shape
  `add` 38 → 114 GB/s; `relu` 94 → 129 GB/s; `softmax [1024,4096]` 11.0 →
  2.1 ms; `broadcast_to().contiguous()` of 4M elements 3.4 → 0.15 ms;
  matmul 1024³ 280 → 310 GFLOP/s.

### Fixed

- **Segfault:** `to_device(Metal)` on a zero-element tensor read 4 bytes
  past an empty allocation (`upload` inflated the copy length along with
  the placeholder size). Allocated, never copied, now; the parity suites
  round-trip `[0,3]`, `[2,0]`, `[0]`, `[2,0,5]` (#17).
- Reductions over a zero extent panicked with index-out-of-bounds; they
  return the identity (sum 0, max −∞, min +∞; mean is NaN). `argmax` over an
  empty dimension returned a fabricated 0 and is now `InvalidArgument`
  (#18).
- `from_vec_f32`/`from_vec_i64` take `impl Into<Shape>` like every other
  constructor (#19).
- `backward()` seeded the gradient on the CPU regardless of the loss's
  device, so backward on a Metal- or CUDA-resident loss failed at the first
  VJP with `DeviceMismatch`. Found by the CUDA end-to-end test; the same
  test now runs on Metal.
- CPU sums are Neumaier-compensated across eight lanes: on a
  cancellation-heavy 100k-element input the serial fold was 0.30 off the
  f64 truth, the new path 0.015.

### Security

- **Shape validation:** `Shape::numel` wrapped on overflow, so a shape like
  `[2^32, 2^32]` had "0" elements, passed the length check against an empty
  buffer, and produced a tensor claiming 2^64 elements over no storage.
  `Shape::checked_numel` gates every caller-supplied shape
  (`from_vec_*`, `reshape`, `broadcast_to`/`broadcast_view`) with a typed
  error, and `numel` itself fails loudly instead of wrapping.
- Unsafe audit: the workspace's `unsafe` is confined to the Metal and CUDA
  backends (FFI-adjacent, each block with a `// SAFETY:` note) and the two
  `Send`/`Sync` impls in storage; eight crates keep
  `#![forbid(unsafe_code)]`. Untrusted-input paths (safetensors load, TOML
  fixtures/replays) validate dtype and shape before allocating.

### Changed

- Reductions, elementwise ops and strided gathers have contiguous fast
  paths (row-based, op dispatched once per call into a monomorphized loop);
  softmax/log_softmax no longer materialize their broadcast max and sum.
  GEMM processes four output rows per task.
- The research crate is `research/oxmera-cuda-oxide` (was `oxmera-cuda`);
  the gate, research workflow and justfile follow. Its verdicts are
  unchanged (strict 0/0/0 at cc 7.5 and 8.6, 12/12 admitted).
- Docs: README, ARCHITECTURE, AGENTS, CONTRIBUTING, LIMITATIONS and ROADMAP
  describe the two CUDA paths; the `cuda deferred` doctor line is gone.

### Earlier in this cycle (post-0.1.1, pre-0.2.0)

- `research/oxmera-cuda-oxide` (then `research/oxmera-cuda`) gains an on-device parity harness
  (`src/bin/parity.rs`, behind the `hardware` feature, run with
  `cargo oxide run --features hardware -- --bin parity`): every kernel
  against an f64 host reference at edge shapes — sizes 1 … 1,000,003,
  under- and over-provisioned grid-stride launches, mismatched lengths,
  reductions at `TILE ± 1`, an all-negative padded max window, persistent
  GEMM grids and a short-output guard. Kernel modules are now `pub` so the
  generated loaders are reachable.

### Measured

- First on-device run, NVIDIA A10G (sm_86, driver 595.71.05, CUDA 13.2):
  **283 parity cases, 0 failures**; Compute Sanitizer memcheck 0 errors,
  racecheck 0 hazards, synccheck 0 errors (quick mode, 227 cases). The
  kernels are unchanged from 0.1.1; correctness only, no timings.

## [0.1.1] — 2026-08-26

Toolchain bump and verification release; no framework API changes.

### Added

- `research/oxmera-cuda-oxide` (then `research/oxmera-cuda`; issue #10): production cuda-oxide kernels —
  grid-stride elementwise ops, staged warp-synchronous tree reductions,
  and double-buffered shared-memory tiled GEMM — statically verified by
  `reconverge check --strict` (0 findings, both compute capabilities) and
  a fully-admitted `launchbound prune` space, enforced in CI by the
  revived `gate.yml` with no GPU anywhere.

### Changed

- **Pins:** reconverge 0.3.0 → 0.4.0 and launchbound 1.2.0 → 2.0.0
  (action `@v2`, SHA-pinned), moved together per the pin policy; nightly,
  cuda-oxide and termlens unchanged. 0.4.0 fixes vyncint/reconverge#65
  (named-const `SharedArray` sizes now counted by RC004) — verified here:
  an 80 KiB named-const tile is `2 deny` and launchbound refuses all four
  of its candidates, where the previous pair admitted them.
- `vec_sigmoid`/`vec_gelu` use plain-arithmetic `exp`/`tanh` instead of
  cuda-oxide's `ex2.approx`/`tanh.approx` intrinsics, which the catalog
  marks `sm_80+`: the crate did not lower for `sm_75` even though
  `kernel.toml` promised `needs_cc = "7.5"`. PTX for both `sm_75` and
  `sm_86` now assembles under `ptxas` (CUDA 13.2). Host-side accuracy
  tests bound the helpers (`exp` ≤ 4e-7 relative; sigmoid/GELU within
  1e-5 of `std`); `just research` runs them.
- `docs/research-baseline.md` re-measured on the new pair.
- termlens 0.6 → 0.6.1; both PTY golden suites and their 100-iteration
  stresses pass unchanged.
- The kernel crate is standalone (own `[workspace]`) so launchbound's
  per-candidate scratch copies resolve; the old `research/` virtual
  workspace manifest is gone.

### Fixed upstream

- vyncint/reconverge#65 (RC004 blind to named-const `SharedArray` sizes,
  found while building the gate) is fixed in reconverge 0.4.0 and
  verified here as described above.

### Known upstream

- vyncint/launchbound#32: `prune --cc` cannot see instruction
  availability, so `needs_cc` is taken on trust; filed with the
  reproduction above.

### Not done

- No kernel has executed on NVIDIA hardware yet; runtime parity and
  timings are tier-2 work and are not claimed.

## [0.1.0] — 2026-08-22

The pivot release (ADR-0006): oxmera is now a functioning tensor and
deep-learning framework. Everything below is new behaviour; 0.0.x was a
deliberate skeleton.

### Added

- **Tensors:** f32 strided storage with zero-copy views (`reshape`,
  `permute`, `transpose`, `narrow`, `slice`, `broadcast_to`, `unsqueeze`),
  NumPy broadcasting, constructors (`zeros`, `ones`, `full`, `randn`,
  `from_slice`, `from_vec_f32`, `from_vec_i64`), element access, and
  `std::ops` operator overloading for tensors and scalars.
- **CPU backend:** rayon-parallel elementwise ops (11 unary, 9 binary with
  broadcasting), axis reductions, argmax, cache-blocked rank-2 GEMM and
  batched rank-3 matmul, gather/scatter.
- **Apple Metal backend:** MSL compute pipelines for strided elementwise,
  axis reductions, threadgroup-memory full reductions, and 16×16 tiled
  GEMM over unified-memory buffers; `to_device` moves tensors both ways.
  CPU↔Metal parity asserted at 1e-5 in on-hardware tests.
- **Autograd:** tape-based reverse mode — `requires_grad`, `backward()`,
  gradient accumulation, `detach`, `zero_grad`, `no_grad` RAII guard —
  with exact VJPs for every primitive and finite-difference gradcheck
  coverage in CI.
- **oxmera-nn:** `Module` trait with shared `Param` handles; `Linear`,
  `Conv2d` (differentiable im2col), `Embedding`, `LayerNorm`,
  `BatchNorm2d`, `Dropout`, `Sequential`; `MSELoss`, `CrossEntropyLoss`,
  `BCEWithLogitsLoss`; Kaiming/Xavier initializers; safetensors
  save/load.
- **oxmera-optim:** `SGD` (momentum + weight decay), `Adam`, `AdamW`,
  `RMSprop` — each proven to converge in tests.
- **CLI:** `oxmera doctor` now reports Apple-Silicon hardware (chip, P/E
  cores, unified memory, Metal budget) and framework capabilities;
  `oxmera train` trains a demo model on CPU or Metal with a plain reporter
  or the `--tui` ratatui dashboard (loss/accuracy sparklines, epoch/batch
  gauges, throughput, memory) — all golden-tested through a real PTY with
  100-iteration determinism stress, driven by a deterministic replay mode.
- `examples/train_mnist.rs` (`--device cpu|metal`), with a synthetic
  offline fallback dataset.

### Changed

- MSRV raised to 1.88 (measured; forced by ratatui).
- The exercise ladder was retired; its specs became the integration test
  suites (ADR-0006).
- The op traits moved into `oxmera-tensor` (orphan-rule requirement for
  operator overloading); `oxmera-ops` re-exports the vocabulary; the CPU
  backend implementation lives in the hub and registers lazily.


## [0.0.3] — 2026-08-22

Still a skeleton (see 0.0.1); no change to any crate's API or behaviour.

### Changed

- Tool pins bumped, one per commit, each re-measured: reconverge
  0.2.0 → **0.3.0**, launchbound 1.0.2 → **1.2.0** (whose internal
  pairing now matches this project's exactly), termlens 0.5 → **0.6**.
  Nightly and cuda-oxide unchanged — reconverge 0.3.0 keeps
  `nightly-2026-04-03` verified against `a766fc26`.
- `docs/research-baseline.md` replaced with measurements under the new
  set: pairing verdicts byte-for-byte identical in both directions
  (clean space admitted, known flip refused with RC001); doctor goldens
  changed only where the fixture version strings did.

## [0.0.2] — 2026-08-21

Still a skeleton (see 0.0.1); no functional change to any crate. This
release exists to prove the tokenless pipeline.

### Changed

- Publishing now uses crates.io Trusted Publishing: the release workflow
  mints a short-lived token from GitHub OIDC per run. The one-time
  first-publish token was deleted and revoked (ADR-0005).
- README badges, `AGENTS.md` as the canonical agent contract (with
  `CLAUDE.md` importing it), refreshed status prose, and a pins watcher
  that fingerprints upstream drift instead of re-reporting it weekly.

## [0.0.1] — 2026-08-21

A name reservation with documentation, not a usable library (ADR-0005).
Every crate is a skeleton: real seams, `todo!()` bodies, no working
operations. `oxmera doctor` (in `oxmera-cli`) is the one thing that runs.
See `docs/LIMITATIONS.md` before depending on anything.

### Added

- Public repository bootstrap: stable root workspace (`oxmera` umbrella
  crate, MSRV 1.85 measured), nightly `research/` workspace
  (`nightly-2026-04-03`, `oxmera-cuda` placeholder), dependency-firewall
  bans in `deny.toml`, `just ci` local gate, governance documents. No
  functionality — everything is a skeleton by design.
- The layer seams: `oxmera-core`, `-tensor`, `-ops`, `-runtime`, `-cpu`,
  reserved `-autograd`/`-nn`/`-optim`, all bodies `todo!()`; ADRs
  0001–0005 (Proposed); per-layer architecture contract.
- The exercise ladder: manifest, harness, tier-A spec crates (A1–A5) and
  tier-B kernel skeletons (B1–B3) plus the written rung B4; `just
  exercise <id>` / `just exercises`.
- `oxmera-cli` with `oxmera doctor`: fixture-driven environment report,
  termlens goldens for three environment shapes, 100-iteration stress.
- CI: matrix ci, MSRV-vs-lockfile, dependency firewall (proven by a
  deliberate violation at gate O2), nightly research check, launchbound
  convergence gate over the tier-B kernels, termlens stress, full-history
  attribution/DCO scan, and a scheduled pins watcher that reports drift
  and never bumps.
- `docs/research-baseline.md`: measured toolchain baselines, including
  the first verification that launchbound 1.0.2 drives reconverge 0.2.0.
