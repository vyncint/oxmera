# ADR-0001 — Crate and module decomposition

Status: Proposed
Date: 2026-08-21

## Context

The layer boundaries are fixed by ARCHITECTURE.md (core / tensor / ops /
runtime / cpu / metal / cli, plus the nightly research workspace). What is
open is naming, granularity, and the re-export strategy.

## Options

### 1. Flat `oxmera-*` crates, one per layer (scaffolded)

One crate per architectural layer, prefixed `oxmera-`, with an `oxmera`
umbrella that re-exports layers as modules (`oxmera::core`,
`oxmera::tensor`, …) plus the common types at the root.

- (+) The dependency firewall is enforceable at the Cargo graph level —
  `cargo deny` bans work per crate, and an illegal dependency is a
  manifest-visible event, not a module import buried in a diff.
- (+) Layer boundaries are compiler-checked: `oxmera-core` physically
  cannot reach tensor code.
- (−) More manifests to version and publish; crates.io rate limits make
  the first release slower.
- (−) Cross-layer refactors touch multiple crates.

### 2. One `oxmera` crate with modules per layer

- (+) Single manifest, fastest publish, easiest refactors.
- (−) The firewall becomes a convention: nothing stops `core` code from
  importing backend modules. The project's own rule is that a convention
  is not an enforcement.

### 3. Two crates: `oxmera` (all stable layers) + `oxmera-cuda`

- (+) Fewer manifests than option 1 while keeping nightly isolation.
- (−) Same internal-firewall weakness as option 2 among the stable layers.

## Decision (proposed)

Option 1, already scaffolded. The firewall being mechanically enforceable
is the deciding property; the publish cost is paid once per release and is
automated. Re-export strategy: the umbrella re-exports each layer crate as
a module and the everyday types (`Tensor`, `Shape`, `DType`, `Device`,
`Error`, `Result`, `TensorOps`) at the root. Switching to option 3 later is
a mechanical merge; switching to option 2 is discouraged for the reason
above.
