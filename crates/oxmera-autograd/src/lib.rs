//! Reverse-mode automatic differentiation for oxmera — reserved layer.
//!
//! Nothing lives here yet, deliberately: the autograd design (tape vs.
//! graph, where gradients attach to [`oxmera_core::Device`]-resident
//! storage) deserves its own ADR once the tensor and op seams have been
//! exercised, and the implementation is the maintainer's to write. This
//! crate exists now so the layer's place in the dependency firewall is
//! fixed from the start: like every stable-workspace crate, it may never
//! depend on `cuda-oxide`, `reconverge`, or `launchbound`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
