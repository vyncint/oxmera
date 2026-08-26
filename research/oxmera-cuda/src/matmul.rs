//! Tiled GEMM with double-buffered shared-memory tiles.
//!
//! Persistent-block form: a 1-D grid where block `g` computes output
//! tiles `g, g + gridDim, …` — the tile loop's trip count is uniform
//! *within* each block (it depends only on `blockIdx`), which is what
//! makes every `sync_threads()` inside it block-uniform, and what
//! `reconverge` verifies.
//!
//! Double buffering: while the block computes on tile buffer `cur`, it
//! prefetches the next K-tile into the other buffer; one barrier per
//! K-step both publishes the prefetch and retires the buffer being
//! swapped away. Edge tiles load zeros, so no branch ever guards a
//! barrier.
//!
//! Dimensions (`M`, `K`, `N`) and the tile edge (`MT`: 16 or 32) are
//! compile-time parameters in `params.rs`; the block is `MT × MT`
//! threads, flattened to a 1-D launch.

use cuda_device::{
    DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

use crate::params::{K, M, MM_THREADS, MT, MT2, N};

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[M, N] = a[M, K] x b[K, N]`, row-major, f32.
    #[kernel]
    #[launch_bounds(MM_THREADS)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn matmul_tiled(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        static mut A_TILES: SharedArray<f32, { 2 * MT2 }> = SharedArray::UNINIT;
        static mut B_TILES: SharedArray<f32, { 2 * MT2 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let r = tid / MT; // row within the tile
        let c = tid % MT; // column within the tile

        let tiles_m = M.div_ceil(MT);
        let tiles_n = N.div_ceil(MT);
        let tiles_k = K.div_ceil(MT);
        let dst = out.as_mut_ptr();

        // Persistent blocks: uniform per block, so everything inside is
        // block-convergent.
        let mut t = thread::blockIdx_x() as usize;
        let grid = thread::gridDim_x() as usize;
        while t < tiles_m * tiles_n {
            let tile_row = (t / tiles_n) * MT;
            let tile_col = (t % tiles_n) * MT;

            // Preload K-tile 0 into buffer 0 (zero-padded at the edges).
            let a0 = load_a(a, tile_row + r, c);
            let b0 = load_b(b, r, tile_col + c);
            // SAFETY: thread (r, c) writes exactly slot r*MT+c of each
            // buffer — disjoint within the block; the barrier publishes
            // the writes before any thread reads them.
            unsafe {
                A_TILES[r * MT + c] = a0;
                B_TILES[r * MT + c] = b0;
            }
            thread::sync_threads();

            let mut acc = 0.0f32;
            let mut cur = 0usize;
            let mut kt = 0usize;
            while kt < tiles_k {
                // Prefetch the next K-tile into the idle buffer. The
                // branch holds no barrier; a straggling prefetch is
                // published by the barrier below.
                let nxt = 1 - cur;
                if kt + 1 < tiles_k {
                    let ka = (kt + 1) * MT;
                    let av = load_a(a, tile_row + r, ka + c);
                    let bv = load_b(b, ka + r, tile_col + c);
                    // SAFETY: same disjoint (r, c) slot argument as the
                    // preload, in the buffer no thread reads this step.
                    unsafe {
                        A_TILES[nxt * MT2 + r * MT + c] = av;
                        B_TILES[nxt * MT2 + r * MT + c] = bv;
                    }
                }

                // Compute on the current buffer.
                let mut kk = 0usize;
                while kk < MT {
                    // SAFETY: reads only, of a buffer fully published by
                    // the barrier that ended the previous step.
                    unsafe {
                        acc += A_TILES[cur * MT2 + r * MT + kk] * B_TILES[cur * MT2 + kk * MT + c];
                    }
                    kk += 1;
                }

                // One uniform barrier per K-step: publishes the prefetch
                // and retires `cur` before it becomes the next prefetch
                // target.
                thread::sync_threads();
                cur = nxt;
                kt += 1;
            }

            let row = tile_row + r;
            let col = tile_col + c;
            if row < M && col < N && row * N + col < out.len() {
                // SAFETY: (row, col) is unique per (t, r, c): tiles are
                // disjoint across `t` (each block owns t ≡ blockIdx mod
                // grid) and (r, c) is unique within the block; the index
                // is bounds-checked against out.
                unsafe { *dst.add(row * N + col) = acc };
            }
            t += grid;
        }
    }
}

/// `a[row, col]`, zero when out of range (edge tiles).
#[inline(always)]
fn load_a(a: &[f32], row: usize, col: usize) -> f32 {
    if row < M && col < K && row * K + col < a.len() {
        a[row * K + col]
    } else {
        0.0
    }
}

/// `b[row, col]`, zero when out of range (edge tiles).
#[inline(always)]
fn load_b(b: &[f32], row: usize, col: usize) -> f32 {
    if row < K && col < N && row * N + col < b.len() {
        b[row * N + col]
    } else {
        0.0
    }
}
