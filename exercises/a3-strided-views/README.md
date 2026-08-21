# A3 — strided views

**Tier A · $0 · gated by:** `cargo test` (this crate's spec)

## What you are learning

That a tensor is a *layout over storage*, and most reshaping is arithmetic
on the layout, not movement of bytes: `permute`, `narrow`, and contiguous
`reshape` are views; `contiguous()` is the one explicit copy. The spec
checks storage identity with `Arc::ptr_eq` — sharing is the point, not an
optimization.

## Where the work is

`crates/oxmera-tensor`: `Tensor::reshape`, `permute`, `narrow`,
`contiguous`, plus whatever A1 constructors are still open. Requires A1.

## What "done" looks like

```bash
just exercise a3
```

Views share storage; `contiguous()` copies exactly when it must and never
otherwise; every error is typed. Then flip `status = "solved"` here and in
the manifest, and unignore the tensor harness test.

## Rules

Expected values are hand-computed from `[[1,2,3],[4,5,6]]`. The spec never
walks a strided buffer itself — iterating a non-contiguous view in logical
order is precisely your exercise.
