# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `research/oxmera-cuda` gains an on-device parity harness
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

- `research/oxmera-cuda` (issue #10): production cuda-oxide kernels —
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
