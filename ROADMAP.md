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
- [x] PTX for `sm_75` and `sm_86` assembled with `ptxas` in the tier-1
      container (0.1.1); every kernel free of `sm_80+`-only intrinsics so
      `needs_cc = "7.5"` holds. Gate pair: reconverge 0.4.0 / launchbound
      2.0.0 (RC004 named-const fix verified; launchbound#32 filed for the
      unchecked `needs_cc`).
- [ ] Host runtime integration (`Device::Cuda` backend) — needs the
      cuda-oxide host crates and a Linux toolchain; kernels are ready.
- [x] Tier-2 correctness on real hardware: parity harness 283/283 and
      Compute Sanitizer (memcheck, racecheck, synccheck) clean on an
      A10G, sm_86 — kernels unchanged from 0.1.1.
- [ ] Tier-2 timings (metered sessions, evidence logs); no performance
      claims until then. cc 7.5 (T4) execution not yet exercised.
