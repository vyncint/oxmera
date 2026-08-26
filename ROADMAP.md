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

## Deferred

- [ ] CUDA backend via `cuda-oxide`, gated by `reconverge` and tuned by
      `launchbound` — the `research/` workspace and the four-pin policy
      exist for this; it returns when the pairing is measured end to end
      on real hardware.
