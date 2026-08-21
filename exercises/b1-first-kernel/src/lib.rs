//! Exercise B1 — the first kernel.
//!
//! An elementwise addition as a cuda-oxide `#[kernel]`: ordinary Rust,
//! compiled to PTX, with the launch contract declared on the function.
//! See README.md for what "done" looks like; the body is yours to write.
//!
//! No barrier belongs in this kernel — each thread touches exactly one
//! element — and noticing *why* is part of the exercise.

mod params;

use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};
use params::LB_MAX;

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = a[i] + b[i]` for every `i` covered by the launch.
    ///
    /// Launch contract: 1-D domain, one thread per element. A thread whose
    /// global index falls outside `out` does nothing.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        // exercise B1: your first kernel. `thread::index_1d()` is the
        // coordinate the contract hands you; `out.get_mut(...)` is the
        // bounds-checked write.
        let _ = (a, b, &mut out, thread::index_1d());
        todo!("exercise B1: implement the elementwise add")
    }
}
