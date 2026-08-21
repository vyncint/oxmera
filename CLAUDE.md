# CLAUDE.md — working agreement for agents and contributors

This file is the contract for any AI agent (and any human using one)
working in this repository.

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
on `cuda-oxide`, `reconverge`, or `launchbound`. `deny.toml` enforces it.
Never add a dependency to a firewalled crate. `cuda-oxide` belongs only in
`research/`; `reconverge` and `launchbound` are tools, never dependencies.

## Attribution

Zero AI attribution anywhere, ever: no `Co-Authored-By` naming an AI or bot,
no "Generated with …", no 🤖 — in commits, PRs, comments, or docs. Commits
are `git commit -sS` (DCO sign-off + signature), Conventional Commits style.

## The local loop

```bash
just ci    # fmt, clippy -D warnings, check, test, cargo-deny, research check
```

Gate on the exit code — `just ci && git commit -sS …` — never on piped
output. A fresh clone must pass with no GPU, no CUDA toolkit, and no LLVM.
The root workspace is stable Rust; `research/` pins its own nightly.

## Never do here

- Publish to crates.io, tag a release, or push to `main` without review.
- Bump `nightly`, `cuda-oxide`, `reconverge`, or `launchbound` — the four
  pins move together, one bump per commit, maintainer-approved.
- Make a GPU, CUDA SDK, or Metal required for `cargo check`/`build`/`test`
  of the default feature set.
- Report an estimate as a measurement, or a local Metal/CPU timing as
  predictive of NVIDIA hardware.
- Commit personal notes, budgets, evidence logs, or scratch files.
