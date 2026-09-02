//! Elementwise kernels: grid-stride loops with structured, reducible
//! control flow and no barriers.
//!
//! Every kernel walks `i = global_thread, i += grid_size` — thread `t`
//! only ever touches indices congruent to `t` modulo the grid size, so
//! writes are disjoint by construction; that argument is repeated at each
//! `unsafe` write.

use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};

use crate::params::LB_MAX;

/// `e^x` in plain f32 arithmetic — deliberately no PTX intrinsic.
///
/// Measured while assembling this crate's PTX in the tier-1 container:
/// every entry in cuda-oxide's float catalog at rev `a766fc26`
/// (`ex2.approx`, `tanh.approx`, `lg2.approx`, `rcp.approx`) is marked
/// `sm_80+`, and `cargo oxide inspect --arch sm_75` refuses to lower
/// them — which would silently contradict kernel.toml's `needs_cc =
/// "7.5"` (the static gate cannot see instruction availability). Plain
/// arithmetic lowers everywhere.
///
/// Cody–Waite range reduction to `|r| <= ln2 / 2`, a degree-6 Taylor
/// polynomial (truncation error `r^7 / 5040 < 1.3e-7`; measured worst
/// relative error against `f32::exp` is 2.53e-7, about 2 ulp), and
/// `2^n` built directly in the exponent field. Input is clamped to
/// `[-87, 88]`, where the result stays a normal f32; outside it `exp`
/// has already saturated for every consumer here.
#[inline(always)]
fn expf(x: f32) -> f32 {
    let x = if x > 88.0 {
        88.0
    } else if x < -87.0 {
        -87.0
    } else {
        x
    };
    let t = x * core::f32::consts::LOG2_E;
    let n = (if t >= 0.0 { t + 0.5 } else { t - 0.5 }) as i32;
    let nf = n as f32;
    // ln 2 split hi/lo so `nf * LN2_HI` is exact in f32.
    let r = x - nf * 0.693_145_75 - nf * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    // n is in [-126, 127] after the clamp, so this is a normal power of two.
    let scale = f32::from_bits(((n + 127) as u32) << 23);
    p * scale
}

/// `tanh(u)` via [`expf`]: `1 - 2 / (e^{2u} + 1)`, clamped where tanh has
/// saturated to ±1 in f32. Absolute error is ≈1e-7 across the range,
/// which is what GELU's `1 + tanh(u)` consumes.
#[inline(always)]
fn tanhf(u: f32) -> f32 {
    // Past |u| = 10, 2 / (e^{2u} + 1) < 4.2e-9 — under half an ulp of 1,
    // so the result is exactly ±1, matching `f32::tanh`.
    let u = if u > 10.0 {
        10.0
    } else if u < -10.0 {
        -10.0
    } else {
        u
    };
    1.0 - 2.0 / (expf(2.0 * u) + 1.0)
}

/// A macro would obscure what reconverge analyzes; the eight kernels are
/// written out so each one's control flow is exactly what is on the page.
// The host-side loader/launcher types `#[cuda_module]` generates carry
// no docs of their own.
#[allow(missing_docs)]
#[cuda_module]
pub mod kernels {
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
            let y = 0.5 * x * (1.0 + tanhf(u));
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

/// Host-side accuracy tests for the portable transcendental helpers. They
/// run under `cargo test` on any machine — the kernels themselves need a
/// device, but `expf`/`tanhf` are plain arithmetic and their error budget
/// is what the GPU-vs-CPU parity tolerance (`1e-5`) rests on.
#[cfg(test)]
mod tests {
    use super::{expf, tanhf};

    fn sweep(lo: f32, hi: f32, n: u32) -> impl Iterator<Item = f32> {
        (0..=n).map(move |i| lo + (hi - lo) * (i as f32) / (n as f32))
    }

    #[test]
    fn expf_relative_error_within_four_ulp_over_the_normal_range() {
        let mut worst = 0.0f32;
        for x in sweep(-87.0, 88.0, 700_000) {
            let got = expf(x);
            let want = x.exp();
            let rel = ((got - want) / want).abs();
            worst = worst.max(rel);
        }
        // Measured worst case 2.53e-7 (≈2.1 ulp); 4e-7 leaves margin for other targets.
        assert!(worst <= 4.0e-7, "worst relative error {worst:e}");
    }

    #[test]
    fn expf_saturates_instead_of_overflowing() {
        assert!(expf(1000.0).is_finite());
        assert!(expf(f32::MAX).is_finite());
        assert!(expf(-1000.0) > 0.0 && expf(-1000.0) < 1e-37);
        assert_eq!(expf(0.0), 1.0);
    }

    #[test]
    fn tanhf_absolute_error_within_gelu_budget() {
        let mut worst = 0.0f32;
        for u in sweep(-12.0, 12.0, 480_000) {
            worst = worst.max((tanhf(u) - u.tanh()).abs());
        }
        assert!(worst <= 2.5e-7, "worst absolute error {worst:e}");
        assert_eq!(tanhf(0.0), 0.0);
        assert_eq!(tanhf(50.0), 1.0);
        assert_eq!(tanhf(-50.0), -1.0);
    }

    #[test]
    fn sigmoid_and_gelu_formulas_match_std_within_parity_tolerance() {
        let mut worst_sig = 0.0f32;
        let mut worst_gelu = 0.0f32;
        for x in sweep(-20.0, 20.0, 400_000) {
            let sig = 1.0 / (1.0 + expf(-x));
            let sig_ref = 1.0 / (1.0 + (-x).exp());
            worst_sig = worst_sig.max((sig - sig_ref).abs());
            let u = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            let gelu = 0.5 * x * (1.0 + tanhf(u));
            let gelu_ref = 0.5 * x * (1.0 + u.tanh());
            worst_gelu = worst_gelu.max((gelu - gelu_ref).abs());
        }
        assert!(worst_sig <= 1e-5, "sigmoid worst abs error {worst_sig:e}");
        assert!(worst_gelu <= 1e-5, "gelu worst abs error {worst_gelu:e}");
    }
}
