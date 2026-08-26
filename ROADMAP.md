# Roadmap

Shipped in 0.1.0: CPU (rayon) + Apple Metal backends, strided views and
broadcasting, batched matmul, reverse-mode autograd (gradcheck-verified),
`Linear`/`Conv2d`/`Embedding`/`LayerNorm`/`BatchNorm2d`/`Dropout`, the
standard losses and optimizers, safetensors weights, and the PTY-tested
`oxmera doctor` + `oxmera train --tui` dashboard.

## Next

- [ ] Metal throughput: batched command encoders, buffer pooling, MPS
      matmul option — with measured before/after numbers
- [ ] Gather/scatter kernels on Metal (drop the CPU round-trip)
- [ ] `f16`/`bf16` storage and compute on Metal
- [ ] Higher-rank matmul broadcasting and `einsum`
- [ ] Data loading utilities and a dataset trait
- [ ] Model zoo examples: CNN on MNIST/CIFAR, char-level transformer
- [ ] TUI: multi-run comparison view, gradient-norm panel

## CUDA (in progress under `research/`)

- [x] Kernel set in `research/oxmera-cuda` (issue #10): grid-stride
      elementwise (add/sub/mul/div/neg/relu/sigmoid/gelu), two-stage
      staged + warp-butterfly reductions (sum/max), and double-buffered
      tiled GEMM (16 and 32 tile edges) — `cargo reconverge check
      --strict` clean and `launchbound prune` fully admitted at cc 7.5
      and 8.6, enforced in CI by `gate.yml` on plain runners.
- [ ] Host runtime integration (`Device::Cuda` backend) — needs the
      cuda-oxide host crates and a Linux toolchain; kernels are ready.
- [ ] Tier-2 measurement on real hardware (metered sessions, evidence
      logs); no timing claims until then.
