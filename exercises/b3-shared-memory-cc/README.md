# B3 — shared memory and `--cc`

**Tier B · $0 · no GPU · gated by:** `launchbound prune` at two compute
capabilities.

## What you are learning

That a verdict is *per-part*. Shared memory per block is a hardware budget
that differs between a T4 (`sm_75`, 64 KiB) and an A10G (`sm_86`, ~99 KiB
usable), so the same kernel with the same tile is refused on one and
admitted on the other — RC004, before any hardware is touched. Timings
differ per part for the same underlying reason; this rung teaches it where
it is cheap.

## Where the work is

`src/lib.rs`: implement the tiled copy (stage → `sync_threads` → write
back), with the barrier discipline from B2.

## What "done" looks like

```bash
cd exercises/b3-shared-memory-cc
cargo reconverge check --strict      # clean: the source has no divergence bug
launchbound prune --cc 8.6 .         # both tiles admitted
launchbound prune --cc 7.5 .         # tile=20480 refused: RC004
```

Both prune outputs read and understood — then flip `status = "solved"`
here and in the manifest. Exercise B4 asks you to explain the refusal in
writing.

## Rules

Do not "fix" the 80 KiB tile by shrinking the space until every part
agrees; the disagreement is the lesson. Mark any kernel that genuinely
*needs* a big-smem part with `needs_cc` instead.
