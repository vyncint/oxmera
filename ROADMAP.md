# Roadmap

Shipped in 0.1.0: CPU (rayon) + Apple Metal backends, strided views and
broadcasting, batched matmul, reverse-mode autograd (gradcheck-verified),
`Linear`/`Conv2d`/`Embedding`/`LayerNorm`/`BatchNorm2d`/`Dropout`, the
standard losses and optimizers, safetensors weights, and the PTY-tested
`oxmera doctor` + `oxmera train --tui` dashboard.

Shipped in 0.2.0: the NVIDIA CUDA backend (`oxmera-cuda`, driver API via
`cudarc`, PTX shipped in the crate, parity- and sanitizer-verified on an
A10G), matmul batch broadcasting, contiguous fast paths that made CPU
reductions 4–44× faster, and the edge-case hardening milestone (#17–#22).

Shipped in 0.3.0 (the "what oxmega needs" milestone, #25–#30): native
gather/scatter on Metal and CUDA, asynchronous Metal dispatch (2.6× on the
motivating benchmark) and a fused Adam/AdamW step, small batched linear
algebra (`cholesky`/`logdet`/`eigh`, differentiable Cholesky), rank-4+
matmul broadcasting and `einsum`, per-group optimizer hyper-parameters,
and `f64` tensors on the CPU.

## Next

- [ ] Metal throughput: buffer pooling, MPS matmul option — with measured
      before/after numbers (async dispatch and the fused optimizer step
      shipped in 0.3.0)
- [ ] `f16`/`bf16` storage and compute on Metal
- [ ] `f64` on the GPU backends (CPU-only in 0.3.0)
- [ ] CUDA throughput: cuBLAS option, multi-stream overlap, buffer pooling,
      `f16` — with measured before/after numbers; measure the 0.3.0
      kernels on the motivating benchmark first
- [ ] `argmax` on the device (still a host round-trip)
- [ ] General linear algebra: LU/solve, SVD, differentiable `eigh`
- [ ] Data loading utilities and a dataset trait
- [ ] Model zoo examples: CNN on MNIST/CIFAR, char-level transformer
- [ ] TUI: multi-run comparison view, gradient-norm panel

## CUDA backend (shipped in 0.2.0, `crates/oxmera-cuda`)

- [x] `Device::Cuda` through the CUDA driver API: strided unary/binary,
      axis + two-stage full reductions, 16×16 tiled matmul with batch
      broadcasting; PTX for `compute_75` embedded, NVRTC rebuild fallback;
      `libcuda` loaded at runtime so nothing needs a toolkit to build.
- [x] Measured on an NVIDIA A10G (sm_86, driver 595.71.05): parity suite
      8/8 at `1e-5`, full workspace tests green with CUDA registered at
      load time, Compute Sanitizer memcheck/racecheck/synccheck clean,
      `oxmera train --device cuda` trains with the same loss trajectory as
      CPU/Metal.
- [ ] Timings (metered sessions, evidence logs); no performance claims
      until then.

## CUDA research line (`research/oxmera-cuda-oxide`)

- [x] Kernel set in `research/oxmera-cuda-oxide` (issue #10): grid-stride
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
- [ ] Host runtime for the cuda-oxide kernels — superseded for shipping
      purposes by `oxmera-cuda`; the research kernels stay a verification
      subject, not a backend.
- [x] Tier-2 correctness on real hardware: parity harness 283/283 and
      Compute Sanitizer (memcheck, racecheck, synccheck) clean on an
      A10G, sm_86 — kernels unchanged from 0.1.1.
- [ ] Tier-2 timings (metered sessions, evidence logs); no performance
      claims until then. cc 7.5 (T4) execution not yet exercised.
