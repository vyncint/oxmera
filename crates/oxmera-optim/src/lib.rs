//! Optimizers for oxmera — reserved layer.
//!
//! Nothing lives here yet, deliberately: optimizers presuppose autograd,
//! which presupposes the tensor rungs. This crate exists now so the
//! layer's place in the dependency firewall is fixed from the start: like
//! every stable-workspace crate, it may never depend on `cuda-oxide`,
//! `reconverge`, or `launchbound` (see [`oxmera_core`] for the vocabulary
//! it will build on).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
