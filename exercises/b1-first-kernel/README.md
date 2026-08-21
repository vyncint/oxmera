# B1 — the first kernel

**Tier B · $0 · no GPU · gated by:** `cargo reconverge check --strict` and
`cargo oxide inspect` (tier 1 container)

## What you are learning

That a CUDA kernel can be ordinary Rust: `#[kernel]` on a function, a
declared launch contract, `cargo oxide` to PTX — no `.cu` file, no GPU in
the room. And the first structural fact of the SIMT model: an elementwise
kernel needs *no barrier*, because no thread ever waits on another.

## Where the work is

`src/lib.rs` in this crate: implement `vec_add` — one thread, one element,
bounds-checked by the contract types (`thread::index_1d()`,
`DisjointSlice::get_mut`).

## What "done" looks like

```bash
cd exercises/b1-first-kernel
cargo reconverge check --strict     # 0 findings — and you know *why* it's trivially clean
launchbound prune --cc 7.5 .        # 3 clean, 0 refused
# tier 1 (container): cargo oxide inspect — read the PTX your Rust became
```

Then flip `status = "solved"` in `exercise.toml` and the manifest.

## Rules

The skeleton declares the contract; the body is yours. Do not add a
barrier to see what happens — that curiosity is exercise B2, where being
caught is the point.
