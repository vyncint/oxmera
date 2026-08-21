# Limitations

This file exists so nobody has to discover any of this the hard way. It is
required reading before depending on anything here, and it is updated
before every release. Numbers, where they exist, live in
[research-baseline.md](research-baseline.md) with the commands that
produced them.

## The big one

**oxmera implements nothing yet.** Every published crate is a skeleton:
real types, real trait signatures, real error taxonomy, and `todo!()`
where every computation would be. Calling any operation panics. This is by
design — the implementations are the maintainer's exercise ladder — and
version `0.0.x` means exactly this. The first version where the CPU
reference backend computes real answers will be `0.1.0`, and no earlier
version claims otherwise.

## Per component

- **The CPU backend is ground truth, and deliberately slow.** When it
  exists, it defines correctness for every other backend and is never
  optimized at the cost of readability. No performance claim is made
  anywhere in this project until a measured one exists, and every measured
  number has an evidence log behind it in a repository you cannot see
  (the maintainer's private lab).
- **The Metal path has no convergence gate**, and never will unless
  someone builds an MSL analyzer. Metal results are validated against the
  CPU reference for correctness, but nothing checks their barrier
  discipline statically. `oxmera doctor` repeats this banner; so does
  every Metal exercise.
- **reconverge's guarantee is inherited wholesale, limits included:**
  summary-based interprocedural analysis, reducible control flow only,
  non-literal masks not evaluable, data races out of scope. **A clean gate
  is not a proof of correctness** — it is the absence of the specific bug
  classes the tool models.
- **cuda-oxide is alpha.** Its API and its codegen move; oxmera pins it by
  revision and treats every bump as a measured event, but alpha is alpha.
- **GPU results do not port between parts.** A safety verdict is
  per-compute-capability (shared-memory context differs between `sm_75`
  and `sm_86`), and timings do not transfer either — launchbound publishes
  that its results do not port even between those two parts, and that its
  no-GPU analytical model's measured Spearman correlation spans 0.00–0.94
  across kernels.
- **Local timings are not predictive of NVIDIA.** CPU and Metal numbers on
  a laptop are useful for relative regression detection on that laptop,
  and for nothing else.

## Process limitations

- Tier-D (real NVIDIA hardware) exercises are rationed by a real budget;
  the interesting GPU lessons arrive in batches, not continuously. Tiers
  A–C are designed to stand alone.
- The exercise ladder can rot: CI compiles unsolved rungs and re-verifies
  solved ones, but nothing can catch a spec that is merely wrong. Specs
  are reviewed on every seam change instead.
