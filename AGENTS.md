# AGENTS.md — working agreement for AI agents and contributors

This file is the canonical contract for any AI agent (and any human using
one) working in this repository. Tool-specific entry points (CLAUDE.md)
import it verbatim; edit the rules here and only here.

## Project status: full production development

oxmera is a Rust tensor and deep-learning framework targeting multi-threaded
CPU, Apple-Silicon Metal and NVIDIA CUDA, with a terminal UI tested through
real PTYs.
The original learning-project boundary — which reserved all computational
code to the maintainer — was **repealed by the maintainer on 2026-08-22**
(ADR-0006). Agents and contributors may implement tensor operations,
kernels, autograd, layers, optimizers, and everything else, at production
quality. The shipped CUDA backend (`oxmera-cuda`) uses the driver API via
`cudarc`; the separate research line (`research/oxmera-cuda-oxide`) is
where `cuda-oxide` kernels are verified with `reconverge` and `launchbound`.

## Ground rules (unchanged)

- **Firewall:** no crate in the stable workspace may depend — directly or
  transitively — on the `cuda-oxide`, `reconverge`, or `launchbound`
  families. `deny.toml` enforces it; those are the research CUDA
  toolchain, never dependencies. (`cudarc` is not in those families and is
  how the shipped backend reaches the driver.)
- **Attribution:** zero AI attribution anywhere, ever — no AI co-author
  trailers, no "Generated with …", no robot emoji, in commits, PRs,
  comments, or docs. Commits are `git commit -sS` (DCO sign-off +
  signature), Conventional Commits style. CI scans full history.
- **Local gate:** `just ci` — judge it by exit code
  (`just ci && git commit -sS …`), never by piped output. A fresh clone
  must build and test with no CUDA toolkit and no LLVM; Metal code is
  `cfg`-gated to macOS, CUDA loads `libcuda` at runtime and registers
  nothing without it, and CI stays green on Linux runners with no GPU.
- **Unsafe policy:** `#![forbid(unsafe_code)]` stays on crates that don't
  need it. Metal/CUDA/FFI code isolates `unsafe` blocks behind safe
  wrappers, each with a `// SAFETY:` explanation; a kernel launch's
  argument list is checked against the kernel signature at the call site.
- **Honesty:** never report an estimate as a measurement; performance
  claims require a re-runnable command and its real output.
- **Hardware:** never provision or bill cloud GPU time; never make a GPU
  required for `cargo check`/`build`/`test` of the default feature set.
- **Main is protected:** changes land by PR with all required checks
  green; releases are tagged by the maintainer only.
