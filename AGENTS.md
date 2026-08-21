# AGENTS.md — working agreement for AI agents and contributors

This file is the canonical contract for any AI agent (and any human using
one) working in this repository. Tool-specific entry points (CLAUDE.md)
import it verbatim; edit the rules here and only here.

## The strict boundary — read this before writing code

The computational parts of oxmera are the **maintainer's to write**. That is
the point of the project. Do not implement:

- tensor operations or mathematical operators
- CUDA or Metal kernels
- autograd, neural-network layers, optimizers
- kernel optimizations or GPU algorithms
- worked examples, demos, or tutorial solutions
- solutions to any exercise under `exercises/`

You may — and should — write: trait definitions and type signatures with
`todo!()` bodies; struct/enum declarations with fields and no logic; error
types, module structure, re-exports, feature flags; tests that specify
behaviour (properties, invariants, hand-computed constant cases); and every
piece of infrastructure — CI, release, tooling, docs, the exercise harness.

**The test-writing rule:** an exercise test may never contain a working
implementation of the thing under test. If a test would let someone pass the
exercise by copying from it, it is the wrong test. If you find yourself
writing a loop over tensor elements, stop.

## The dependency firewall

No crate in the stable root workspace may depend — directly or transitively —
on the `cuda-oxide`, `reconverge`, or `launchbound` families. `deny.toml`
enforces it (exact-name bans plus `unknown-git`) and CI fails on violations.
Never add a dependency to a firewalled crate. `cuda-oxide` belongs only in
`research/` and the tier-B exercises; `reconverge` and `launchbound` are
tools, never dependencies.

## The pin policy

Nightly (`nightly-2026-04-03`), reconverge, cuda-oxide, and launchbound move
together or not at all — one bump per commit, no behaviour change riding
along, maintainer-approved. CONTRIBUTING.md names every file a bump must
touch.

## Attribution

Zero AI attribution anywhere, ever: no `Co-Authored-By` naming an AI or bot,
no "Generated with …", no robot emoji — in commits, PRs, comments, or docs.
Commits are `git commit -sS` (DCO sign-off + signature), Conventional
Commits style. CI scans full history.

## The local loop

```bash
just ci             # fmt, clippy -D warnings, check, test, docs, cargo-deny, research check
just exercises      # the ladder: compiles todo rungs, verifies solved ones
```

Gate on the exit code — `just ci && git commit -sS …` — never on piped
output. A fresh clone must pass with no GPU, no CUDA toolkit, and no LLVM.
The root workspace is stable Rust; `research/` and the tier-B exercises pin
their own nightly.

## Never do here

- Publish to crates.io, tag a release, or push to `main` without review —
  `main` is protected; changes land by PR with all required checks green.
- Bump any of the four pins without maintainer approval.
- Make a GPU, CUDA SDK, or Metal required for `cargo check`/`build`/`test`
  of the default feature set.
- Provision or bill cloud GPU time.
- Report an estimate as a measurement, or a local Metal/CPU timing as
  predictive of NVIDIA hardware.
- Commit personal notes, budgets, evidence logs, or scratch files.
