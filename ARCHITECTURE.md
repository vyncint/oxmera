# Architecture

Status: this file describes the intended shape. The layer crates land at
stage O1 (see [ROADMAP.md](ROADMAP.md)); when they do, each layer's section
here gains its three answers — what it owns, what it must never know about,
and which layers may depend on it.

## Layers

```
oxmera            umbrella re-export, no logic
oxmera-core       dtype, shape, strides, device handle, error taxonomy
oxmera-tensor     tensor type, storage ownership, views
oxmera-ops        operation traits — signatures only
oxmera-runtime    dispatch and scheduling seam
oxmera-cpu        reference backend: correct, unoptimized, always available
oxmera-metal      local execution on Apple Silicon (feature-gated, macOS)
oxmera-cli        `oxmera doctor` and friends; termlens-tested
--- separate nightly workspace: research/ ---
oxmera-cuda       cuda-oxide path (feature-gated, Linux, non-default)
exercises/*       the exercise ladder
```

## The dependency firewall

`core`, `tensor`, `ops`, `runtime`, `cpu`, and every future autograd / NN /
optimizer crate have **zero** dependency — direct or transitive — on
`cuda-oxide`, `launchbound`, or `reconverge`. This is enforced mechanically:
`deny.toml` bans all three families, and CI fails on a deliberate violation.
A convention is not an enforcement.

## Two workspaces, and why

oxmera core targets **stable Rust**. `cuda-oxide` and `reconverge` require a
pinned nightly. Rather than dragging the whole project onto nightly, the
GPU/compiler research layer lives in a separate workspace at `research/`
with its own `rust-toolchain.toml`. (ADR-0003, landing at O1, records the
alternatives.)

Consequence, designed for deliberately: `cargo check --workspace
--all-targets` in the root must succeed on `aarch64-apple-darwin` with no
CUDA toolkit, no LLVM, and no network access to NVIDIA anything. If a fresh
clone cannot check on a MacBook, the setup has failed.

## Two orthogonal concerns, never merged

- **Correctness on a laptop** → `cargo reconverge check --strict`, then
  `launchbound prune`. Full guarantee, no GPU, no cost. This is where kernel
  bugs are found, and it is the whole reason laptop-only GPU learning works.
- **Performance on a laptop** → Metal or the CPU backend. **No convergence
  gate exists on the Metal path.** Useful only for relative regression
  detection within one machine.

Local numbers are **not** predictive of NVIDIA. launchbound publishes that
its results do not port even between `sm_75` and `sm_86`, and that its
no-GPU analytical model's measured Spearman correlation spans 0.00–0.94
across kernels. A safety verdict is also per-part: shared-memory context
(RC004) depends on `--cc`, so a verdict at one compute capability does not
transfer to another.

## The four external tools

| tool | pin | role |
|---|---|---|
| `cuda-oxide` | rev `a766fc26` | rustc codegen backend, Rust → PTX. Kernels are ordinary Rust marked `#[kernel]`. Research workspace only. Cannot build on macOS. |
| `reconverge` | 0.2.0 | compile-time static verifier for divergent barriers and non-convergent warp ops. Runs as a wrapped `cargo check` — no PTX, no GPU. The correctness gate reachable from a laptop. |
| `launchbound` | 1.0.2, action `@v1` | convergence-safe autotuner. CLI and CI tool driven from the justfile and GitHub Actions — **never a Cargo dependency**. `prune` needs no GPU. |
| `termlens` | 0.5.0 | PTY-driving terminal test harness for every terminal surface, starting with `oxmera doctor`. |

The four pins (nightly `nightly-2026-04-03`, reconverge, cuda-oxide,
launchbound) move together or not at all. See CONTRIBUTING.md for the bump
policy and every file a bump must touch.
