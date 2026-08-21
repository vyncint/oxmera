# A2 — broadcasting

**Tier A · $0 · gated by:** `cargo test` (this crate's spec)

## What you are learning

The shape-compatibility rules every elementwise operation depends on,
written as a *total function*: every pair of shapes has a defined answer —
a result shape or a typed refusal carrying the operands. No panics, no
special cases discovered later in a kernel.

## Where the work is

`crates/oxmera-core`: `shape::broadcast_shapes`. (Requires A1's `Shape`
work.)

## What "done" looks like

```bash
just exercise a2
```

Every hand case broadcasts (or refuses) exactly as NumPy would; the
properties hold: commutative, idempotent, scalar-identity, and the result
dominates both operands. Then flip `status = "solved"` here and in the
manifest.

## Rules

The spec never computes a broadcast result — expected shapes are written
by hand. If a property test fails, minimize the counterexample by hand and
understand it before touching the code.
