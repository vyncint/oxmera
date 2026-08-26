# ADR-0006 — The pivot to full implementation

Status: Accepted
Date: 2026-08-22

## Context

oxmera was founded as a public learning workbench with a strict boundary:
agents and contributors built the seams, specs, and tooling, and the
computational parts — tensor ops, kernels, autograd, layers, optimizers —
were reserved for the maintainer to implement by hand, rung by rung, as an
exercise ladder. That boundary was the project's charter, stated in the
governance files and enforced socially from v0.0.1 through v0.0.3.

## Decision

On 2026-08-22 the maintainer repealed the boundary, with the consequences
laid out explicitly and confirmed: the repository is open for full
production development, AI-assisted implementation included, and the
learning-project premise publicly ends. The exercise ladder's specs were
converted into the integration test suites; the rungs themselves were
retired.

What did **not** change:

- the dependency firewall (`cuda-oxide`/`reconverge`/`launchbound` are
  never dependencies of the stable workspace) — CUDA is deferred, not
  abandoned, and its tools stay tools;
- commit hygiene: DCO + signatures, Conventional Commits, zero AI
  attribution in history (the maintainer's standing AI-tooling policy:
  what lands is signed off as their own work);
- measurement honesty: no estimate reported as a measurement, no
  performance claim without a re-runnable command.

## Consequences

- Versions from 0.1.0 describe a functioning framework, per ADR-0005's
  version discipline ("0.1.0 = the first version where the CPU backend
  computes real answers").
- The public history shows the skeleton-to-framework transition plainly;
  this ADR is the honest marker of when and why.
- `docs/research-baseline.md` and the pins table remain, serving the
  deferred CUDA path under `research/`.
