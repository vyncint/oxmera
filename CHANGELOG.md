# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
