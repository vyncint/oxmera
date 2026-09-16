# Stability

What oxmera 0.5 promises, what may still change before 1.0, and how the
promise is enforced. This is the contract; the running record of what
actually changed is [CHANGELOG.md](../CHANGELOG.md).

## SemVer, pre-1.0

oxmera follows [Semantic Versioning](https://semver.org). While the version
is `0.x`, the pre-1.0 rules apply:

- A **minor** bump (`0.4 → 0.5`) may contain breaking changes.
- A **patch** bump (`0.5.0 → 0.5.1`) may not: it is bug fixes and additive,
  backward-compatible changes only.

Every crate in the workspace shares one version and is released together,
so "the version" is unambiguous.

This is enforced, not just stated: the `semver-checks` CI job runs
[`cargo-semver-checks`](https://crates.io/crates/cargo-semver-checks)
against the last release on crates.io on every pull request. A breaking
change in a patch fails CI; a breaking change under a minor bump is
allowed and expected.

Breaking changes are also called out by hand: each one is a
`**Breaking.**` bullet in the CHANGELOG for the release that ships it, with
the one-line migration.

## What 0.5 stabilizes

The surfaces a typical program touches are meant to be stable for the rest
of the `0.5.x` series:

- `Tensor` and its construction, view, elementwise, reduction, matmul and
  linear-algebra methods, and the `oxmera_core::Error` taxonomy.
- The `nn` layers (`Linear`, `Conv2d`, `Embedding`, `LayerNorm`,
  `BatchNorm2d`, `Dropout`, `Sequential`), the `Module` trait, losses, and
  safetensors save/load by parameter name.
- The optimizers (`Sgd`, `Adam`, `AdamW`, `RmsProp`) and `ParamGroup`.
- The `oxmera` umbrella crate's re-exports and `oxmera::init()`.
- The `oxmera` CLI's `doctor` and `train` surfaces, including
  `doctor --json`.

Several public types are `#[non_exhaustive]`, so adding a variant or field
to them is *not* a breaking change: `Error`, `Device`, the op enums
(`UnaryOp`, `BinaryOp`, `ReduceOp`), `MatmulPlan`, `AdamStep`, and
`ParamGroup`. Match them with a `_ =>` arm and construct them through their
constructors.

## What may still change before 1.0

- **Backends.** The `Backend` trait and the `AdamStep`/`MatmulPlan`
  plumbing are how the tensor layer reaches a device. They are public so an
  out-of-tree backend is possible, but they are the surface most likely to
  grow methods before 1.0.
- **Multi-GPU.** Only device index 0 is registered today; `Device::Cuda`/
  `Device::Metal` with `index > 0` is a typed `BackendUnavailable`. How
  multiple devices are selected and registered is not settled
  ([docs/LIMITATIONS.md](LIMITATIONS.md)).
- **The seed convention.** Layer constructors take a positional `u64`
  seed; whether that becomes an explicit RNG handle is open.

## MSRV

The minimum supported Rust version is **1.88**, declared as
`rust-version` and checked in CI against the committed `Cargo.lock`. Raising
the MSRV is a change worth a CHANGELOG note but is not by itself a breaking
change under these pre-1.0 rules; it will not happen in a patch release.

## Supported versions

Per [SECURITY.md](../SECURITY.md), only the latest published version of
each crate is supported. Fixes land on `main` and ship in the next release
rather than being backported.

## Toward 1.0

1.0 is reached when the surfaces above have gone a full release cycle with
no breaking change, the backend trait is settled, and the multi-GPU and
seed questions are decided one way or the other. The roadmap tracks
progress: [ROADMAP.md](../ROADMAP.md).
