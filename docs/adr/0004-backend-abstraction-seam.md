# ADR-0004 — The backend abstraction seam

Status: Proposed
Date: 2026-08-21

## Context

This is the decision that shapes everything written next: where `Device`,
`Storage`, and `Backend` sit, and whether dispatch is static or dynamic.
It is presented as a genuine question — the scaffold follows the option
judged strongest, but the alternatives are kept cheap to switch to and the
maintainer decides.

## Options

### 1. Dynamic dispatch: `Device` enum + `Arc<dyn Backend>` (scaffolded)

`Tensor` is one concrete type holding `Arc<Storage>`; the runtime resolves
`Device -> Arc<dyn Backend>`; op traits are object-safe.

- (+) One `Tensor` type in every signature, error message, and exercise —
  the gentlest surface for a learning codebase.
- (+) Devices are runtime values: `oxmera doctor` and mixed-device
  programs need no generics gymnastics.
- (+) Backends are open: a new backend crate registers itself without
  touching core types.
- (−) A vtable call per op — irrelevant here, where ops move whole
  tensors, and this project refuses performance claims anyway.
- (−) Per-dtype typing is deferred to runtime checks (`DTypeMismatch`
  errors rather than compile errors).

### 2. Static dispatch: `Tensor<B: Backend>` (the burn shape)

- (+) Illegal cross-backend mixing is a compile error; zero dispatch cost.
- (−) Every signature in every exercise carries `<B: Backend>`; validating
  Metal against the CPU reference — the project's core loop — now involves
  two distinct `Tensor` types and explicit conversion plumbing.
- (−) Trait bounds compound as op families grow (the burn experience).

### 3. Closed enum dispatch: `Storage` enum with a variant per backend
(the candle shape)

- (+) No vtables, still one `Tensor` type.
- (−) The backend set is closed: adding Metal or CUDA edits the core enum,
  which the dependency firewall makes awkward (core must not know GPU
  crates exist — the variants would have to hold opaque boxes, which is
  option 1 wearing a costume).

## Decision (proposed)

Option 1, scaffolded: `Device` (handle) in `oxmera-core`, `Storage` and
`Tensor` in `oxmera-tensor`, op traits in `oxmera-ops`, `Backend` +
registry + `TensorOps` in `oxmera-runtime`. The op traits are kept
object-safe as a standing constraint (the ops harness test asserts it), so
option 3 remains a mechanical rewrite and option 2 a larger but bounded
one. Revisit when autograd lands — tape designs sometimes prefer static
backends — and record the outcome here.
