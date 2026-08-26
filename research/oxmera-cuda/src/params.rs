//! Compile-time tuning parameters. Repo defaults; `launchbound` rewrites
//! this file per candidate in a scratch copy of the crate, never in the
//! repository.

/// Shared-memory stage length (elements) for the reductions — dimension
/// `tile` in kernel.toml. One block stages `TILE` elements and reduces
/// them. `TILE * 4 + 128` bytes of static shared memory must stay within
/// the 48 KiB static cap (RC004). reconverge 0.4.0 evaluates this named
/// const (vyncint/reconverge#65, fixed there); at 20480 it fires RC004.
pub const TILE: usize = 1024;

/// `#[launch_bounds]` max threads for the 1-D kernels. Must cover every
/// `block_x` value in kernel.toml.
pub const LB_MAX: u32 = 256;

/// Matrix tile edge for `matmul_tiled` (16 or 32): the block is
/// `MT x MT` threads and each shared tile is `MT * MT` floats.
pub const MT: usize = 16;

/// Elements in one matrix tile.
pub const MT2: usize = MT * MT;

/// Threads per matmul block (`MT * MT`), as the launch bound.
pub const MM_THREADS: u32 = (MT * MT) as u32;

/// Compile-time GEMM extents: `[M, K] x [K, N] -> [M, N]`. Benchmark
/// sizes; divisible by both supported tile edges.
pub const M: usize = 128;
/// Shared dimension.
pub const K: usize = 128;
/// Output columns.
pub const N: usize = 128;
