//! Exercise B3 — shared memory and `--cc`.
//!
//! The launch space in kernel.toml includes a tile that fits an A10G's
//! shared-memory budget (`--cc 8.6`) but not a T4's (`--cc 7.5`). Nothing
//! in this source file is wrong; one *configuration* of it is impossible
//! on one *part*. The lesson: a safety verdict is per-compute-capability,
//! and so is everything else — RC004 is just the first place you see it.

mod params;

use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};
use params::{LB_MAX, TILE};

#[cuda_module]
mod kernels {
    use super::*;

    /// Tiled copy through shared memory: stage a tile, synchronize, write
    /// it back out. Deliberately simple — the tile *size* is the subject.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn tile_copy(data: &[f32], mut out: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<f32, TILE> = SharedArray::UNINIT;

        // exercise B3: stage, sync_threads, write back. Keep the barrier
        // discipline you learned in B2; the interesting part is running
        //   launchbound prune --cc 7.5 .   vs   launchbound prune --cc 8.6 .
        let _ = (data, &mut out, &raw const SMEM, thread::index_1d());
        todo!("exercise B3: implement the tiled copy")
    }
}
