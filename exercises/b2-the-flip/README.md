# B2 — the flip

**Tier B · $0 · no GPU · gated by:** `cargo reconverge check --strict` —
first by *failing* it, then by passing it.

## What you are learning

The single most important fact about SIMT correctness: a barrier's safety
is a property of the *launch configuration*, not just the source. The same
`if warp::warp_id() == 0 { sync_threads() }` is fine at one warp and
undefined at two — and on real hardware the broken version often runs, and
sometimes runs *faster*. You cannot see this class of bug by testing. A
static gate can.

## The two-phase exercise

1. Implement the reduction with the guarded barrier. Run
   `cargo reconverge check --strict`. **Read the finding**: RC001, with
   the span pointing into your `src/lib.rs`. This failure is the deliverable
   of phase one.
2. Fix the discipline — every thread reaches every barrier — and run the
   gate again: clean at `--strict`, and `launchbound prune --cc 7.5 .`
   admits both block sizes.

## What "done" looks like

The fixed kernel in `src/lib.rs`; the strict gate clean; one sentence in
this README (below) on why the guard was tempting. Then flip
`status = "solved"` here and in the manifest.

## Why the guard was tempting (write yours after being caught)

> _…_
