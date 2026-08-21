# A5 — the error taxonomy

**Tier A · $0 · gated by:** `cargo test` (this crate's spec)

## What you are learning

Error *design*, not error handling: making the invalid states
unrepresentable (a `Shape` cannot hold a negative extent; a `Device` is a
closed set) and making every representable failure a typed value that
carries what the caller needs — the operands, the operation, the bounds.
This taxonomy is the language every later backend speaks when it refuses.

## Where the work is

`crates/oxmera-core`: `Device::kind_name`, and auditing the `Error` enum
as you solve A1–A4 — if a failure path forced you to invent a stringly
error or reuse a wrong variant, the taxonomy (not the call site) is what
needs fixing. Extending the enum is part of the exercise; the spec pins
the contracts, not the variant count.

## What "done" looks like

```bash
just exercise a5
```

Every refusal names its operation and carries its operands; `Error` stays
`std::error::Error + Send + Sync`. Then flip `status = "solved"` here and
in the manifest.

## Rules

The spec asserts what messages must *contain*, never their exact phrasing
— wording is yours. No `unwrap()` in library code paths; `todo!()`
disappears from `oxmera-core` entirely when this rung closes.
