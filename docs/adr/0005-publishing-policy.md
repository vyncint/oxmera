# ADR-0005 — Publishing policy

Status: Proposed
Date: 2026-08-21

## Context

Publishing skeleton crates is a name reservation. That is legitimate when
the description says so and indefensible when it implies capability. This
ADR decides what a `0.0.x` release means, what the crates.io metadata must
say, and when a version may first claim behaviour.

## Options

### 1. Publish `0.0.1` skeletons with self-describing metadata (scaffolded)

Every crate's `description` states "skeleton under construction: no working
operations yet" (already true in the manifests). `docs/LIMITATIONS.md` and
the README limitations section exist before the release. The version stays
`0.0.x` while any core seam body is `todo!()`.

- (+) Reserves the names honestly; makes the learning project publicly
  followable from crates.io, not just GitHub.
- (+) Proves the release machinery end to end while the cost of a botched
  release is near zero.
- (−) crates.io accumulates skeleton crates; some readers will not read
  the description.

### 2. Do not publish until something works

- (+) Nothing on crates.io can be misread.
- (−) The names stay unreserved (squattable), and the release pipeline —
  a stated project deliverable — goes unproven until the worst possible
  moment: the first release that matters.

### 3. Reserve names with empty `0.0.0` stubs, real releases later

- (−) Strictly worse than option 1: same reservation, less honesty (no
  docs, no source of truth), and crates.io discourages bare squatting.

## Decision (proposed)

Option 1. Version discipline: `0.0.x` = skeleton, may claim nothing;
`0.1.0` = the first version where the CPU reference backend computes real
answers for the published op surface, and not before; any performance
claim, ever, requires a committed evidence log. Every version bump is its
own commit.

Release mechanics (recorded here so the deviation is visible): the project
end-state is crates.io Trusted Publishing with no stored registry token.
Trusted Publishing can only be configured for crates that already exist on
crates.io, so the **first** publish of each crate uses a short-lived
`CARGO_REGISTRY_TOKEN` repository secret; immediately after `v0.0.1`, the
trusted publisher is configured for every crate, the secret is deleted,
and the token revoked. From then on the tokenless rule in SECURITY.md
holds unconditionally.

**2026-08-21, completed:** v0.0.1 published all ten crates with the
token; the maintainer then configured trusted publishers, deleted the
secret, and revoked the token. `release.yml` mints per-run tokens via
`rust-lang/crates-io-auth-action` from v0.0.2 onward.
