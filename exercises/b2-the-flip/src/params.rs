//! Compile-time tuning parameters. Repo defaults; the tuner rewrites this
//! file in scratch copies only.

/// Shared-memory tile length (elements). Dimension `tile` in kernel.toml.
pub const TILE: usize = 128;

/// `#[launch_bounds]` max threads. Must cover every `block_x` value in
/// kernel.toml — including the one that flips the verdict.
pub const LB_MAX: u32 = 64;
