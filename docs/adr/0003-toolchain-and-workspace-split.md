# ADR-0003 — Toolchain and workspace split

Status: Proposed
Date: 2026-08-21

## Context

oxmera core targets stable Rust (MSRV measured, currently 1.85).
`cuda-oxide` and `reconverge` require one specific pinned nightly
(`nightly-2026-04-03`), and reconverge is a rustc driver — it and the rustc
it wraps must be the same build. These two worlds must coexist without the
nightly leaking into what a fresh clone needs.

## Options

### 1. Two workspaces: stable root + `research/` on the pinned nightly (scaffolded)

- (+) `cargo check --workspace --all-targets` at the root works on a
  MacBook with no CUDA toolkit, no LLVM, no nightly — the project's
  non-negotiable acceptance property.
- (+) `rust-toolchain.toml` does the pinning per directory; no wrapper
  scripts, no `+toolchain` incantations to forget.
- (−) Two lockfiles, two `cargo` invocations in CI, and cross-workspace
  path dependencies are not allowed (research crates may not depend on
  root crates without publishing or `[patch]` tricks — acceptable, since
  the research layer speaks to the stable layer through published
  interfaces or not at all).

### 2. One workspace, nightly toolchain everywhere

- (+) One lockfile, one graph.
- (−) Every consumer and CI job inherits the nightly; MSRV becomes
  meaningless; a nightly bump risks the whole project instead of one
  workspace. Rejected outright.

### 3. One stable workspace, GPU crates behind non-default features

- (+) One workspace on paper.
- (−) Feature-gating cannot change the *toolchain*; `cargo check
  --all-features` would still demand nightly + LLVM + CUDA. The gate the
  project cares about would fail exactly where it matters.

## Decision (proposed)

Option 1, already scaffolded and exercised (the O0 gate ran against it).
The isolation property is load-bearing for everything else the project
promises.
