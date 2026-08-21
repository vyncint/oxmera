# A1 — shape and strides

**Tier A · $0 · gated by:** `cargo test` (this crate's spec)

## What you are learning

How a tensor's logical index space maps onto flat memory: row-major
layout, element strides, the index→offset arithmetic, and what
"contiguous" actually means. Everything a GPU kernel does with an index
later is this arithmetic; get it into your fingers on the CPU first.

## Where the work is

Replace the `todo!()` bodies in `crates/oxmera-core`:

- `DType::size_in_bytes`, `DType::is_float`
- `Shape::numel`
- `layout::contiguous_strides`
- `Layout::contiguous`, `Layout::offset_of`, `Layout::is_contiguous`
- then in `crates/oxmera-tensor`: `Storage::cpu_zeros`,
  `Storage::cpu_from_bytes`, `Tensor::from_storage`, `Tensor::from_vec_f32`,
  `Tensor::get_f32`

## What "done" looks like

```bash
just exercise a1        # every spec in tests/spec.rs passes
```

Then flip `status = "solved"` in `exercise.toml` and this rung's entry in
`exercises/manifest.toml`, and unignore the harness tests that name A1.

## Rules

The spec asserts hand-computed cases and properties (bijectivity,
unit-stride last dimension). It deliberately never computes an offset for
you, and no solution may be copied into the tests.
