//! Compile-time tuning parameters. Repo defaults; `launchbound tune`
//! rewrites this file per candidate in a scratch copy of the crate, never
//! in the repository.

/// `#[launch_bounds]` max threads (`.maxntid`). Must cover every `block_x`
/// value in kernel.toml.
pub const LB_MAX: u32 = 256;
