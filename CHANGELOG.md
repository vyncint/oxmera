# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
