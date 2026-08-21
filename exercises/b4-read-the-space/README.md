# B4 — read the space

**Tier B · $0 · no GPU · gated by:** your own written explanation
(`ANSWER.md` in this directory; the harness checks it exists and is real).

## What you are learning

To read what an autotuner actually enumerates before you ever pay to run
one. `launchbound space` shows a kernel's launch/specialization space;
`launchbound prune` shows which configurations the safety gate refuses and
under which rule. Tier D's `tune` will rank the survivors on real hardware
— this rung is where you learn to understand its input.

## The exercise

Run, against your solved B2 and B3 kernels:

```bash
launchbound space ../b2-the-flip
launchbound prune --cc 7.5 ../b2-the-flip
launchbound space ../b3-shared-memory-cc
launchbound prune --cc 7.5 ../b3-shared-memory-cc
launchbound prune --cc 8.6 ../b3-shared-memory-cc
```

Then write `ANSWER.md` here, in your own words:

1. For one refused configuration of each kernel: *which* rule refused it,
   and what about that configuration — not the source — triggered it.
2. Why the same B3 configuration gets two different verdicts at the two
   `--cc` values, and what that implies about reusing tuning results
   across parts.

## What "done" looks like

`ANSWER.md` written (a paragraph or two of real explanation), B2 and B3
solved first. Flip `status = "solved"` here and in the manifest.

## Rules

No quoting tool output as the answer — the tool already said it; the rung
is you saying it back in your own words.
