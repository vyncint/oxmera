//! Two-stage reductions: shared-memory staging plus warp-synchronous
//! butterfly reduction.
//!
//! Barrier discipline (the part `reconverge` proves): every
//! `sync_threads()` sits at top level of the kernel body — no barrier is
//! ever under a thread-dependent branch, so all threads of a block reach
//! every barrier uniformly. Warp collectives (`warp::reduce_*_f32`) are
//! full-warp butterflies executed at the top level of the kernel body —
//! never under a thread-derived branch — so every warp that exists is
//! fully convergent at each collective (the first draft guarded the
//! combine with `warp 0`; reconverge flagged it as RC002 and the hoisted
//! form is what it verifies clean).
//!
//! Shape: block `b` reduces the `TILE`-element window starting at
//! `b * TILE` and writes one partial to `out[b]`; the host (or a second
//! launch over the partials) finishes. `TILE` is the tunable
//! shared-memory budget launchbound's RC004 checks per compute
//! capability.

use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};

use crate::params::{LB_MAX, TILE};

/// Warps per block at the largest supported block size (1024 threads).
const MAX_WARPS: usize = 32;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum of block `b`'s `TILE`-wide window into `out[b]`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
        static mut STAGE: SharedArray<f32, TILE> = SharedArray::UNINIT;
        static mut PARTIALS: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bd = thread::blockDim_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let base = block * TILE;

        // Stage the window. The loop bound is uniform across the block;
        // out-of-range elements stage the identity.
        let mut j = tid;
        while j < TILE {
            let g = base + j;
            let v = if g < data.len() { data[g] } else { 0.0 };
            // SAFETY: each thread writes stage slots j ≡ tid (mod bd) —
            // disjoint within the block; the barrier below orders these
            // writes before any cross-thread read.
            unsafe { STAGE[j] = v };
            j += bd;
        }
        thread::sync_threads();

        // Every thread accumulates its strided share of the stage.
        let mut acc = 0.0f32;
        let mut j = tid;
        while j < TILE {
            // SAFETY: reads only, after the staging barrier.
            unsafe { acc += STAGE[j] };
            j += bd;
        }

        // Stage one: full-warp butterfly; every lane holds its warp's sum.
        let wsum = warp::reduce_sum_f32(acc);
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        if lane == 0 {
            // SAFETY: one write per warp, indexed by warp id — disjoint.
            unsafe { PARTIALS[wid] = wsum };
        }
        thread::sync_threads();

        // Stage two, hoisted to top level so the collective is
        // convergent by construction (reconverge flags a full-warp mask
        // under a thread-derived branch, RC002): every warp redundantly
        // butterflies the same per-warp partials, and thread 0 writes.
        let warps = bd.div_ceil(32);
        // SAFETY: reads only, after the partials barrier.
        let v = if lane < warps {
            unsafe { PARTIALS[lane] }
        } else {
            0.0
        };
        let total = warp::reduce_sum_f32(v);
        if tid == 0 && block < out.len() {
            // SAFETY: block indices are unique per block and checked
            // against out.len() — one thread writes one slot.
            unsafe { *out.as_mut_ptr().add(block) = total };
        }
    }

    /// Maximum of block `b`'s `TILE`-wide window into `out[b]`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn reduce_max(data: &[f32], mut out: DisjointSlice<f32>) {
        static mut STAGE: SharedArray<f32, TILE> = SharedArray::UNINIT;
        static mut PARTIALS: SharedArray<f32, MAX_WARPS> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bd = thread::blockDim_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let base = block * TILE;

        let mut j = tid;
        while j < TILE {
            let g = base + j;
            let v = if g < data.len() {
                data[g]
            } else {
                f32::NEG_INFINITY
            };
            // SAFETY: as in `reduce_sum` — disjoint stage writes, ordered
            // by the barrier below.
            unsafe { STAGE[j] = v };
            j += bd;
        }
        thread::sync_threads();

        let mut acc = f32::NEG_INFINITY;
        let mut j = tid;
        while j < TILE {
            // SAFETY: reads only, after the staging barrier.
            let v = unsafe { STAGE[j] };
            if v > acc {
                acc = v;
            }
            j += bd;
        }

        let wmax = warp::reduce_max_f32(acc);
        let lane = warp::lane_id() as usize;
        let wid = tid / 32;
        if lane == 0 {
            // SAFETY: one write per warp, indexed by warp id — disjoint.
            unsafe { PARTIALS[wid] = wmax };
        }
        thread::sync_threads();

        // Hoisted like `reduce_sum`: the collective runs at top level in
        // every warp, convergent by construction; thread 0 writes.
        let warps = bd.div_ceil(32);
        // SAFETY: reads only, after the partials barrier.
        let v = if lane < warps {
            unsafe { PARTIALS[lane] }
        } else {
            f32::NEG_INFINITY
        };
        let total = warp::reduce_max_f32(v);
        if tid == 0 && block < out.len() {
            // SAFETY: as in `reduce_sum` — unique block slot, bounds
            // checked.
            unsafe { *out.as_mut_ptr().add(block) = total };
        }
    }
}
