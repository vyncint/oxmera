# oxmera

[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![crates.io](https://img.shields.io/crates/v/oxmera?label=crates.io)](https://crates.io/crates/oxmera)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-orange)](Cargo.toml)
[![GPU](https://img.shields.io/badge/Apple%20Metal-accelerated-brightgreen)](crates/oxmera-metal)
[![ci](https://img.shields.io/github/actions/workflow/status/vyncint/oxmera/ci.yml?label=ci)](https://github.com/vyncint/oxmera/actions/workflows/ci.yml)
[![stress](https://img.shields.io/github/actions/workflow/status/vyncint/oxmera/stress.yml?label=pty%20stress)](https://github.com/vyncint/oxmera/actions/workflows/stress.yml)

A Rust-native tensor and deep-learning framework: multi-threaded CPU and
**Apple-Silicon Metal** backends, tape-based reverse-mode **autograd**,
neural-network layers and optimizers, **safetensors** weights, and an
interactive **terminal training dashboard** — every terminal surface tested
through a real PTY with deterministic golden frames.

```rust
use oxmera::nn::{CrossEntropyLoss, Linear, Module, Sequential};
use oxmera::optim::{Adam, Optimizer};
use oxmera::{Device, Tensor};

let model = Sequential::new()
    .push(Linear::new(2, 32, 1))
    .push(Linear::new(32, 2, 2));
let mut opt = Adam::new(model.parameters(), 1e-2);

let x = Tensor::randn([64, 2]).to_device(Device::Metal { index: 0 })?;
let logits = model.forward(&x)?;
// … loss.backward()?; opt.step()?;
```

## What's inside

| area | what you get |
|---|---|
| tensors | `f32` strided views (`reshape`/`permute`/`narrow`/`broadcast_to` are zero-copy), NumPy broadcasting, batched matmul, operator overloading (`&a + &b`, `a * 2.0`) |
| devices | CPU (rayon-parallel, cache-tiled GEMM) and Apple Metal (MSL compute kernels, threadgroup reductions, tiled GEMM over unified memory); `tensor.to_device(...)` moves data, autograd flows across the move |
| autograd | tape-based reverse mode: `requires_grad`, `backward()`, gradient accumulation, `no_grad` RAII guard — every VJP validated by finite differences in CI |
| nn | `Linear`, `Conv2d`, `Embedding`, `LayerNorm`, `BatchNorm2d`, `Dropout`, `Sequential`; `MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`; Kaiming/Xavier initializers |
| optim | `SGD` (momentum, weight decay), `Adam`, `AdamW`, `RMSprop` |
| weights | zero-config `safetensors` save/load by parameter name |
| terminal | `oxmera doctor` (hardware, devices, capabilities) and `oxmera train --tui` (live loss/accuracy sparklines, progress gauges, throughput, unified-memory usage) — both golden-tested through a real PTY with a 100-iteration determinism stress |

## Quick start

```bash
cargo add oxmera                     # library
cargo install oxmera-cli             # the `oxmera` binary
oxmera doctor                        # what can this machine do?
oxmera train --tui --device metal    # watch a model train, live
```

Run the example (MNIST if `./data/mnist` holds the IDX files, a synthetic
dataset otherwise):

```bash
cargo run --release -p oxmera --example train_mnist -- --device metal
```

## Correctness, not vibes

- The CPU backend is the reference; the Metal backend is **asserted equal
  to it within `1e-5`** across every op family, in tests that run on real
  Apple-Silicon hardware.
- Every backward pass is checked against **central finite differences**.
- Terminal output is captured from a **real PTY** (via `termlens`) and
  compared to golden frames; a 100-iteration stress proves frame-for-frame
  determinism. No clocks, no absolute paths, no flaky snapshots.
- `cargo clippy -D warnings`, doc-warnings-as-errors, `cargo deny`
  (licenses, bans, advisories), and a measured MSRV (1.88), all in CI.

## What this deliberately is not (yet)

- **No CUDA backend.** Deferred; the research scaffolding for a
  convergence-checked CUDA path (via `cuda-oxide` + `reconverge` +
  `launchbound`) lives under `research/` and will return as a backend
  later. No stable-workspace crate may depend on those tools — CI enforces
  the firewall.
- **`f32`-first.** Integer tensors exist for indices and targets; wider
  dtype coverage is roadmap.
- **No performance claims without measurements.** See
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what is and is not
  promised — including why tiny-batch Metal runs are slower than CPU.

## History

oxmera began as a public learning workbench whose computational parts were
deliberately reserved for the maintainer to implement by hand
([ADR-0006](docs/adr/0006-the-pivot-to-full-implementation.md) records the
pivot to full production development on 2026-08-22). The exercise specs
from that era live on as this repository's integration tests.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option. See [CONTRIBUTING.md](CONTRIBUTING.md): DCO + signed commits,
Conventional Commits, zero AI attribution in history, and a dependency
firewall around the deferred CUDA toolchain.
