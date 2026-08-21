# Roadmap

Two tracks. The **workbench** track is infrastructure — seams, tooling, CI,
release machinery. The **climb** is the maintainer implementing the
computational parts as exercises; those are deliberately not built by the
workbench, and each milestone below is something the maintainer implements
by hand.

## Workbench (infrastructure, in order)

- [x] O0 — public repo bootstrap: workspaces, governance, firewall config, local loop
- [x] O1 — the seams: layer crates with signatures and `todo!()` bodies; ADRs 0001–0005
- [x] O2 — the firewall, proven: a deliberate violation failed CI naming the rule, then was reverted
- [x] O3 — toolchain reconnaissance: baselines measured; launchbound 1.0.2 verified driving reconverge 0.2.0
- [x] O4 — the exercise ladder: harness, manifest, skeletons, specs (tiers A and B; C and D planned)
- [x] O5 — the terminal surface: `oxmera doctor`, termlens-tested from its first commit
- [x] O6 — CI complete on public runners
- [x] O7 — branch protection and security settings
- [ ] O8 — the release pipeline, proven end to end (v0.0.1; token for the first publish per ADR-0005, then Trusted Publishing)

## The climb (implementations, each one an exercise)

1. Shape and strides — index ↔ offset, contiguity, row-major layout
2. Broadcasting — the shape-compatibility rules, as a total function
3. Strided views — iterate a non-contiguous view in logical order
4. The reference matmul — correct, deliberately naive; ground truth for every backend
5. The error taxonomy — invalid states unrepresentable, representable failures typed
6. First CUDA kernel — elementwise `#[kernel]` with a declared launch contract
7. The flip — write a divergent barrier and *be caught* by `reconverge` (RC001)
8. Shared memory and `--cc` — watch a verdict differ per compute capability (RC004)
9. Read the space — `launchbound space` / `prune` on my own kernel
10. Metal elementwise and reduction — validated against the CPU reference
11. NVIDIA compile / run / tune — metered sessions, evidence-logged

Nothing on the climb has a date. Tier-D work is rationed by a real budget
and arrives in batches.
