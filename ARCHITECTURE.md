# Architecture

The layer crates exist as seams — real signatures, `todo!()` bodies. Each
layer answers three questions: what it owns, what it must never know
about, and who may depend on it. Design decisions live in
[docs/adr/](docs/adr/); ADR-0004 (the backend seam) shapes most of what
follows.

## Layers

```
oxmera            umbrella re-export, no logic
oxmera-core       dtype, shape, strides, device handle, error taxonomy
oxmera-tensor     tensor type, storage ownership, views
oxmera-ops        operation traits — signatures only
oxmera-runtime    dispatch and scheduling seam
oxmera-cpu        reference backend: correct, unoptimized, always available
oxmera-autograd   reserved: reverse-mode AD (nothing lives here yet)
oxmera-nn         reserved: layers (nothing lives here yet)
oxmera-optim      reserved: optimizers (nothing lives here yet)
oxmera-metal      local execution on Apple Silicon (feature-gated, macOS; not created yet — ADR-0002)
oxmera-cli        `oxmera doctor` and friends; termlens-tested
--- separate nightly workspace: research/ ---
oxmera-cuda       cuda-oxide path (feature-gated, Linux, non-default)
exercises/*       the exercise ladder
```

### The three answers, per layer

**`oxmera-core`**
- Owns: the vocabulary — `DType`, `Shape`, `Strides`, `Layout`, `Device`
  (a handle, not a backend), the `Error` taxonomy, and the shape/stride/
  broadcast arithmetic (exercise rungs A1/A2/A5).
- Must never know about: tensors, storage, backends, dispatch, any GPU
  toolchain, any I/O.
- Depended on by: everything. Depends on `std` and `thiserror` only.

**`oxmera-tensor`**
- Owns: `Storage` (an owned, refcounted, dtype-tagged buffer on one
  device) and `Tensor` (a `Layout` over shared storage), plus the view
  operations that change layout without touching data (rung A3).
- Must never know about: how any operation computes, which backends
  exist, dispatch policy.
- Depended on by: `ops`, `runtime`, backends, umbrella. Depends on `core`.

**`oxmera-ops`**
- Owns: the operation traits (`UnaryOps`, `BinaryOps`, `MatmulOps`,
  `ReduceOps`) and the documented contracts backends must honor.
  Signatures only, permanently — implementations never live here.
- Must never know about: which backends exist, the runtime, scheduling.
- Depended on by: `runtime` and every backend. Depends on `core`, `tensor`.

**`oxmera-runtime`**
- Owns: the `Backend` trait (op traits + identity), the device→backend
  registry, and the user-facing `TensorOps` surface that dispatches
  through it. Dispatch is dynamic (`Arc<dyn Backend>`) per ADR-0004.
- Must never know about: how operations compute; `cuda-oxide`,
  `reconverge`, `launchbound` (backends register themselves — the runtime
  never reaches for them).
- Depended on by: backends and the umbrella. Depends on `core`, `tensor`,
  `ops`.

**`oxmera-cpu`**
- Owns: the ground truth. Its implementations define what every operation
  means; every other backend is validated against it; it is never
  optimized at the cost of readability.
- Must never know about: other backends, dispatch policy, anything
  GPU-shaped.
- Depended on by: the umbrella only (and, for validation, backend test
  suites). Depends on `core`, `tensor`, `ops`, `runtime`.

**`oxmera-autograd` / `oxmera-nn` / `oxmera-optim`** (reserved)
- Own: nothing yet — each is a placeholder fixing the layer's place in
  the firewall before its design ADR exists.
- Must never know about: `cuda-oxide`, `reconverge`, `launchbound` —
  reserved crates are inside the firewall from birth.
- Depended on by: nothing yet. Depend on `core` (and later `tensor`/`ops`).

**`oxmera-cli`**
- Owns: the terminal surface — `oxmera doctor`, its environment probing,
  and its deterministic rendering contract (no clocks, no absolute paths,
  fixture-injectable). Pure infrastructure: the one crate allowed to work
  before the exercises are solved.
- Must never know about: backend internals; it reads public surfaces and
  the exercise manifest, nothing deeper. Its goldens run on fixtures,
  never on probed hardware.
- Depended on by: nobody — it is a binary (`oxmera`). Depends on `serde`
  and `toml`; termlens is dev-only.

**`oxmera` (umbrella)**
- Owns: re-exports only — each layer as a module, the everyday types at
  the root. No logic, permanently.
- Must never know about: anything the layers do not already export.
- Depended on by: users. Depends on every stable layer.

**`oxmera-cuda` (research workspace)**
- Owns: the cuda-oxide kernels and their launch contracts, once tier B
  opens.
- Must never know about: the stable workspace's internals beyond published
  interfaces; it can never be a member of the root workspace (ADR-0003).
- Depended on by: nothing in the stable workspace, ever. That direction is
  the firewall.

## The dependency firewall

`core`, `tensor`, `ops`, `runtime`, `cpu`, and every future autograd / NN /
optimizer crate have **zero** dependency — direct or transitive — on
`cuda-oxide`, `launchbound`, or `reconverge`. This is enforced mechanically:
`deny.toml` bans all three families, and CI fails on a deliberate violation.
A convention is not an enforcement.

## Two workspaces, and why

oxmera core targets **stable Rust**. `cuda-oxide` and `reconverge` require a
pinned nightly. Rather than dragging the whole project onto nightly, the
GPU/compiler research layer lives in a separate workspace at `research/`
with its own `rust-toolchain.toml`. (ADR-0003 records the alternatives.)

Consequence, designed for deliberately: `cargo check --workspace
--all-targets` in the root must succeed on `aarch64-apple-darwin` with no
CUDA toolkit, no LLVM, and no network access to NVIDIA anything. If a fresh
clone cannot check on a MacBook, the setup has failed.

## Two orthogonal concerns, never merged

- **Correctness on a laptop** → `cargo reconverge check --strict`, then
  `launchbound prune`. Full guarantee, no GPU, no cost. This is where kernel
  bugs are found, and it is the whole reason laptop-only GPU learning works.
- **Performance on a laptop** → Metal or the CPU backend. **No convergence
  gate exists on the Metal path.** Useful only for relative regression
  detection within one machine.

Local numbers are **not** predictive of NVIDIA. launchbound publishes that
its results do not port even between `sm_75` and `sm_86`, and that its
no-GPU analytical model's measured Spearman correlation spans 0.00–0.94
across kernels. A safety verdict is also per-part: shared-memory context
(RC004) depends on `--cc`, so a verdict at one compute capability does not
transfer to another.

## The four external tools

| tool | pin | role |
|---|---|---|
| `cuda-oxide` | rev `a766fc26` | rustc codegen backend, Rust → PTX. Kernels are ordinary Rust marked `#[kernel]`. Research workspace only. Cannot build on macOS. |
| `reconverge` | 0.3.0 | compile-time static verifier for divergent barriers and non-convergent warp ops. Runs as a wrapped `cargo check` — no PTX, no GPU. The correctness gate reachable from a laptop. |
| `launchbound` | 1.2.0, action `@v1` | convergence-safe autotuner. CLI and CI tool driven from the justfile and GitHub Actions — **never a Cargo dependency**. `prune` needs no GPU. |
| `termlens` | 0.5.0 | PTY-driving terminal test harness for every terminal surface, starting with `oxmera doctor`. |

The four pins (nightly `nightly-2026-04-03`, reconverge, cuda-oxide,
launchbound) move together or not at all. See CONTRIBUTING.md for the bump
policy and every file a bump must touch.
