# oxmera

A Rust-native machine-learning and GPU-computing framework built from first
principles, in the open, as a learning project that does not lie about what
it is.

**Status: skeleton under construction.** No tensor operation, kernel, or
backend is implemented yet. What exists today is the workspace, the tooling,
and the rules the rest will be built under. Every claim in this README is
either measured or marked as a plan.

## What this is

A first-principles ML/GPU framework built in public as a learning project,
with a convergence-checked CUDA path and an exercise ladder anyone can
follow. Correctness is defined by a deliberately unoptimized CPU reference
backend; every other backend is validated against it. The framework is the
artifact, but the education is the product.

## What this deliberately is not

- **Not a competitor** to [`candle`](https://github.com/huggingface/candle),
  [`burn`](https://github.com/tracel-ai/burn), `tch`, or `dfdx`. Those are
  mature; if you need a framework today, use one of them.
- **Not fast.** The CPU backend is deliberately naive, and no claim of
  performance is made anywhere until a measured one exists.
- **Not a CUDA binding layer.** The CUDA path uses
  [`cuda-oxide`](https://github.com/NVlabs/cuda-oxide), which compiles Rust
  to PTX. There are no `.cu` files anywhere in this project.
- **Not a convergence analyzer or an autotuner.** Those are
  [`reconverge`](https://github.com/vyncint/reconverge) and
  [`launchbound`](https://github.com/vyncint/launchbound), which oxmera
  *uses as tools* and never vendors.
- **Not production-ready**, or ready at all.

## The three backends are not symmetric

| backend | runs where | correctness gate | what its numbers mean |
|---|---|---|---|
| CPU reference | everywhere, stable Rust | it *is* the ground truth — deliberately naive | nothing; it defines correct, not fast |
| Metal | Apple Silicon, locally | validated against the CPU reference; **no convergence gate exists on this path** | relative regression detection within one machine, only |
| CUDA (`cuda-oxide`) | Linux + NVIDIA, non-default feature | `cargo reconverge check --strict` and `launchbound prune`, both **without a GPU** | real timings come only from real hardware, and are per-part |

Local timings are not predictive of NVIDIA. GPU results do not port between
parts — launchbound publishes that its results do not transfer even between
`sm_75` and `sm_86`. This project repeats that instead of assuming otherwise.

## The exercise ladder

The ladder is what makes this a learning workbench rather than a scaffold:
skeletons and specs are built into the repo, and the implementations are the
maintainer's exercise. It lands at stage O4 (see [ROADMAP.md](ROADMAP.md)).

| tier | rungs | needs | cost |
|---|---|---|---|
| A — Rust and the CPU | shape/strides, broadcasting, strided views, reference matmul, error taxonomy | stable Rust | $0 |
| B — GPU correctness with no GPU | first kernel, the flip (be caught by RC001), shared memory and `--cc`, read the space | pinned nightly, reconverge, launchbound | $0 |
| C — Metal | elementwise, reduction | Apple Silicon | $0 |
| D — NVIDIA | compile, run and measure, tune | a metered cloud GPU session | metered |

## The local loop

A fresh clone must work on a machine with **no GPU, no CUDA toolkit, and no
LLVM**. The root workspace is stable Rust (MSRV 1.85, measured against the
lockfile); the `research/` workspace pins `nightly-2026-04-03`.

```bash
just ci     # fmt, clippy -D warnings, check, test, cargo-deny, research check
```

Gate on the exit code, never on output.

## Limitations

Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before depending on
anything here. The short version: **nothing is implemented yet** — every
crate is a published skeleton and every operation panics; the Metal path
has no convergence gate; a clean reconverge gate is not a proof of
correctness; GPU verdicts and timings are per-part; and local timings
predict nothing about NVIDIA hardware.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR —
this project has unusual rules (DCO + signed commits, a strict
learning-boundary policy, and a four-pin toolchain contract).
