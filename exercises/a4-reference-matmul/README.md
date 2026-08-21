# A4 — the reference matmul

**Tier A · $0 · gated by:** `cargo test` (this crate's spec)

## What you are learning

The operation every later backend must reproduce, written for correctness
and nothing else. This implementation becomes the project's ground truth:
Metal (C1/C2) and CUDA (D-tier) results are compared against it, so its
clarity matters more than its speed — three nested loops you can prove
correct beat anything clever.

## Where the work is

`crates/oxmera-cpu`: `CpuBackend::matmul` (and the A1/A3 plumbing it reads
through — note the spec multiplies a *transposed view*, so strided reads
must already work). Wiring `oxmera-runtime` dispatch (`backend_for`,
`register_backend`, the `TensorOps` impl) also belongs to this rung: it is
what makes the umbrella harness test pass end to end.

## What "done" looks like

```bash
just exercise a4
```

Hand-computed products, the identity and zero laws, the transpose law
through views, column independence, typed shape errors. Then flip
`status = "solved"` here and in the manifest, and unignore the cpu,
runtime, and umbrella harness tests.

## Rules

No performance claims — this backend is deliberately naive, and the README
of the repo says so. The spec contains algebraic laws and hand-computed
constants only; the multiplication loop exists nowhere but your
implementation.
