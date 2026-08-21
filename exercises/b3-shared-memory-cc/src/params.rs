//! Compile-time tuning parameters. Repo defaults; the tuner rewrites this
//! file in scratch copies only.

/// Shared-memory tile length (elements). The kernel.toml space includes
/// 20480 (80 KiB of f32) — inside an A10G's per-block budget, past a T4's.
pub const TILE: usize = 4096;

/// `#[launch_bounds]` max threads.
pub const LB_MAX: u32 = 256;
