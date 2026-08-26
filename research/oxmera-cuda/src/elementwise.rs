//! Elementwise kernels: grid-stride loops with structured, reducible
//! control flow and no barriers.
//!
//! Every kernel walks `i = global_thread, i += grid_size` — thread `t`
//! only ever touches indices congruent to `t` modulo the grid size, so
//! writes are disjoint by construction; that argument is repeated at each
//! `unsafe` write.

use cuda_device::{
    DisjointSlice, cuda_module, float, kernel, launch_bounds, launch_contract, thread,
};

use crate::params::LB_MAX;

/// `e^x` from the `2^x` PTX intrinsic.
#[inline(always)]
fn expf(x: f32) -> f32 {
    float::ex2_approx_f32(x * core::f32::consts::LOG2_E)
}

/// A macro would obscure what reconverge analyzes; the eight kernels are
/// written out so each one's control flow is exactly what is on the page.
#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = a[i] + b[i]`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = min3(a.len(), b.len(), out.len());
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            // SAFETY: i < n <= out.len(), and i ≡ start (mod stride) with
            // start unique per thread and start < stride, so no two
            // threads ever write the same element.
            unsafe { *dst.add(i) = a[i] + b[i] };
            i += stride;
        }
    }

    /// `out[i] = a[i] - b[i]`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_sub(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = min3(a.len(), b.len(), out.len());
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = a[i] - b[i] };
            i += stride;
        }
    }

    /// `out[i] = a[i] * b[i]`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_mul(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = min3(a.len(), b.len(), out.len());
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = a[i] * b[i] };
            i += stride;
        }
    }

    /// `out[i] = a[i] / b[i]` (IEEE-754 semantics; division by zero gives
    /// the IEEE result, never a trap).
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_div(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = min3(a.len(), b.len(), out.len());
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = a[i] / b[i] };
            i += stride;
        }
    }

    /// `out[i] = -a[i]`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_neg(a: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = if a.len() < out.len() {
            a.len()
        } else {
            out.len()
        };
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = -a[i] };
            i += stride;
        }
    }

    /// `out[i] = max(a[i], 0)` — branch-free, so every lane takes the
    /// same path.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_relu(a: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = if a.len() < out.len() {
            a.len()
        } else {
            out.len()
        };
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            let x = a[i];
            let y = if x > 0.0 { x } else { 0.0 };
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = y };
            i += stride;
        }
    }

    /// `out[i] = 1 / (1 + e^-a[i])`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_sigmoid(a: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = if a.len() < out.len() {
            a.len()
        } else {
            out.len()
        };
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            let y = 1.0 / (1.0 + expf(-a[i]));
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = y };
            i += stride;
        }
    }

    /// GELU with the tanh approximation, matching the CPU and Metal
    /// backends: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
    #[kernel]
    #[launch_bounds(LB_MAX)]
    #[launch_contract(domain = 1, coordinates = u32)]
    pub fn vec_gelu(a: &[f32], mut out: DisjointSlice<f32>) {
        let start = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as usize;
        let n = if a.len() < out.len() {
            a.len()
        } else {
            out.len()
        };
        let dst = out.as_mut_ptr();
        let mut i = start;
        while i < n {
            let x = a[i];
            let u = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            let y = 0.5 * x * (1.0 + float::tanh_approx_f32(u));
            // SAFETY: as in `vec_add` — in bounds, disjoint by congruence.
            unsafe { *dst.add(i) = y };
            i += stride;
        }
    }
}

fn min3(a: usize, b: usize, c: usize) -> usize {
    let ab = if a < b { a } else { b };
    if ab < c { ab } else { c }
}
