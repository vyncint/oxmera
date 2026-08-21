# AGENTS.md — working agreement for AI coding agents

The full contract lives in [CLAUDE.md](CLAUDE.md); this file is the
tool-agnostic pointer to it, in the emerging `AGENTS.md` convention. If
your agent reads only one file, make it that one. The short version:

## The one rule that defines this project

The computational parts of oxmera — tensor operations, kernels, autograd,
layers, optimizers, and **solutions to any exercise under `exercises/`** —
are the maintainer's to write. That is the point of the project. Never
implement them; build the room instead: seams, types, signatures with
`todo!()` bodies, tests-as-specs, tooling, CI, docs. An exercise test may
assert properties and hand-computed cases; it may never contain a working
implementation of the thing under test.

## Ground rules

- **Firewall:** no stable-workspace crate may depend on `cuda-oxide`,
  `reconverge`, or `launchbound` families — `deny.toml` enforces it and CI
  fails on violations.
- **Pins:** nightly, reconverge, cuda-oxide, and launchbound move together
  or not at all, one bump per commit, maintainer-approved.
  CONTRIBUTING.md names every file a bump must touch.
- **Attribution:** zero AI attribution anywhere — no AI co-author
  trailers, no "Generated with", no robot emoji. Commits are
  `git commit -sS` (DCO + signature), Conventional Commits style. CI
  scans full history.
- **Local gate:** `just ci` — judge it by exit code, never by piped
  output. A fresh clone must pass with no GPU, no CUDA toolkit, no LLVM.
- **Hardware:** never make a GPU or Metal required for check/build/test of
  the default feature set; never provision or bill cloud GPU time; never
  report an estimate as a measurement.
- **Main is protected:** changes land by PR with all required checks
  green; releases are tagged by the maintainer only.
