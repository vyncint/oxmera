//! oxmera — a Rust-native machine-learning and GPU-computing framework built
//! from first principles, in the open, as a learning project that does not
//! lie about what it is.
//!
//! This is the umbrella crate: it re-exports the public surface of the
//! oxmera workspace and contains no logic of its own. The layer crates
//! (`oxmera-core`, `oxmera-tensor`, `oxmera-ops`, `oxmera-runtime`,
//! `oxmera-cpu`, …) land behind this facade; until they do, this crate is
//! intentionally empty.
//!
//! Status: skeleton under construction. No operations are implemented.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
