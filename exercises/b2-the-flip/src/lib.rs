//! Exercise B2 — the flip. **Being caught is the exercise.**
//!
//! You will implement this shared-memory reduction twice:
//!
//! 1. First, the tempting-but-wrong way: guard the barrier —
//!    `if warp::warp_id() == 0 { thread::sync_threads(); }` — and run
//!    `cargo reconverge check --strict`. Read the RC001 finding it hands
//!    you: the rule, the source span, *your* line number. At `block_x=32`
//!    that guard is true for every thread and the kernel is safe; at 64 it
//!    is undefined behaviour. Same source. That is the flip.
//! 2. Then fix it — every thread reaches every barrier, unconditionally —
//!    and watch the finding disappear.
//!
//! The rung is solved when the *fixed* kernel is in this file and the
//! strict gate is clean. The README wants one sentence, in your words, on
//! why the guarded barrier was ever tempting.

mod params;

use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};
use params::{LB_MAX, TILE};

#[cuda_module]
mod kernels {
    use super::*;

    /// Block-level sum of `data` tiles into `out`, staged through shared
    /// memory. The barrier discipline is the entire lesson.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn reduce(data: &[f32], mut out: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<f32, TILE> = SharedArray::UNINIT;

        // exercise B2: stage into SMEM, synchronize, reduce, write out.
        // Write the guarded-barrier version FIRST and let RC001 catch you.
        let _ = (data, &mut out, &raw const SMEM, thread::index_1d());
        todo!("exercise B2: the flip — see the module docs")
    }
}
