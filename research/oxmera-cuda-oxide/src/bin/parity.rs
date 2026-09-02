//! On-device parity harness: every kernel against an f64 host reference,
//! at the shapes that exercise the edges the static gate cannot see.
//!
//! Run on a machine with an NVIDIA GPU:
//! `cargo oxide run --features hardware --bin parity`. Exit code 0 means
//! every case was within tolerance; the log is the evidence.
//!
//! Tolerance is `|gpu - ref| <= 1e-5 * max(1, |ref|)` for elementwise
//! ops and matmul; sums use a relative bound because the GPU's tree
//! summation order differs from a serial fold.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use oxmera_cuda_oxide::params::{K, M, MT, N, TILE};
use oxmera_cuda_oxide::{elementwise, matmul, reduce};

/// Deterministic pseudo-random f32 in `[-scale, scale]` (xorshift, no deps).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self, scale: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (2.0 * u - 1.0) * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

struct Report {
    cases: usize,
    failures: usize,
}

impl Report {
    fn check(&mut self, name: &str, got: &[f32], want: &[f64], tol: impl Fn(usize, f64) -> f64) {
        self.cases += 1;
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        let mut worst = 0.0f64;
        let mut worst_i = 0usize;
        let mut bad = 0usize;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            let err = ((g as f64) - w).abs();
            let lim = tol(i, w);
            if err > worst {
                worst = err;
                worst_i = i;
            }
            if !(err <= lim) {
                bad += 1;
            }
        }
        if bad == 0 {
            println!(
                "  ok   {name:<44} n={:<9} worst|err|={worst:.3e} at [{worst_i}]",
                got.len()
            );
        } else {
            self.failures += 1;
            println!(
                "  FAIL {name:<44} n={:<9} {bad} out of tolerance; worst|err|={worst:.3e} at [{worst_i}] got={} want={}",
                got.len(),
                got[worst_i],
                want[worst_i]
            );
        }
    }
}

fn abs_tol(_i: usize, w: f64) -> f64 {
    1e-5 * w.abs().max(1.0)
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!(
        "device: {} (sm_{major}{minor}); TILE={TILE} MT={MT} M={M} K={K} N={N}",
        ctx.device_name().expect("device name")
    );

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut rep = Report {
        cases: 0,
        failures: 0,
    };

    // ----------------------------------------------------------- elementwise
    // SAFETY: the embedded module was built from this crate by the same compiler invocation.
    let ew = unsafe { elementwise::kernels::load(&ctx) }.expect("load elementwise module");
    // Sizes: 1, a warp-1, odd prime, just past a block, and large; grids:
    // fewer blocks than needed (grid-stride path) and more (idle threads).
    // PARITY_QUICK=1 drops the million-element cases (for Compute Sanitizer,
    // which slows every access by orders of magnitude).
    let quick = std::env::var_os("PARITY_QUICK").is_some();
    let sizes: Vec<usize> = [1usize, 31, 1_009, 257, 65_537, 1_000_003]
        .into_iter()
        .filter(|&n| !quick || n < 100_000)
        .collect();
    let shapes = [(1u32, 32u32), (3, 64), (17, 128), (4_096, 256)];
    for &n in &sizes {
        let a = rng.vec(n, 10.0);
        let b: Vec<f32> = rng
            .vec(n, 10.0)
            .into_iter()
            .map(|x| if x.abs() < 0.5 { 0.75 } else { x })
            .collect();
        let a_d = DeviceBuffer::from_host(&stream, &a).unwrap();
        let b_d = DeviceBuffer::from_host(&stream, &b).unwrap();
        for &(grid, block) in &shapes {
            let cfg = LaunchConfig1D::new(grid, block, 0);
            let tag = format!("g{grid}xb{block}");
            macro_rules! binop {
                ($k:ident, $p:ident, $f:expr) => {{
                    let mut o = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
                    // Any 1-D shape works: the kernels are grid-stride; buffers cover n.
                    let prep = ew.$p(cfg).expect(stringify!($p));
                    ew.$k(&stream, &prep, &a_d, &b_d, &mut o)
                        .expect(stringify!($k));
                    let got = o.to_host_vec(&stream).unwrap();
                    let want: Vec<f64> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| $f(x as f64, y as f64))
                        .collect();
                    rep.check(&format!("{} {tag}", stringify!($k)), &got, &want, abs_tol);
                }};
            }
            macro_rules! unop {
                ($k:ident, $p:ident, $f:expr) => {{
                    let mut o = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
                    // As above.
                    let prep = ew.$p(cfg).expect(stringify!($p));
                    ew.$k(&stream, &prep, &a_d, &mut o).expect(stringify!($k));
                    let got = o.to_host_vec(&stream).unwrap();
                    let want: Vec<f64> = a.iter().map(|&x| $f(x as f64)).collect();
                    rep.check(&format!("{} {tag}", stringify!($k)), &got, &want, abs_tol);
                }};
            }
            binop!(vec_add, prepare_vec_add, |x, y| x + y);
            binop!(vec_sub, prepare_vec_sub, |x, y| x - y);
            binop!(vec_mul, prepare_vec_mul, |x, y| x * y);
            binop!(vec_div, prepare_vec_div, |x, y| x / y);
            unop!(vec_neg, prepare_vec_neg, |x: f64| -x);
            unop!(vec_relu, prepare_vec_relu, |x: f64| x.max(0.0));
            unop!(vec_sigmoid, prepare_vec_sigmoid, |x: f64| 1.0
                / (1.0 + (-x).exp()));
            unop!(vec_gelu, prepare_vec_gelu, |x: f64| 0.5
                * x
                * (1.0
                    + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x))
                        .tanh()));
        }
    }
    // Mismatched lengths: the kernels take min(len) — out longer than inputs
    // must leave the tail untouched (zero), inputs longer must be ignored.
    {
        let a = rng.vec(1_000, 1.0);
        let b = rng.vec(1_500, 1.0);
        let a_d = DeviceBuffer::from_host(&stream, &a).unwrap();
        let b_d = DeviceBuffer::from_host(&stream, &b).unwrap();
        let mut o = DeviceBuffer::<f32>::zeroed(&stream, 1_200).unwrap();
        // As above.
        let prep = ew.prepare_vec_add(LaunchConfig1D::new(8, 128, 0)).unwrap();
        ew.vec_add(&stream, &prep, &a_d, &b_d, &mut o).unwrap();
        let got = o.to_host_vec(&stream).unwrap();
        let want: Vec<f64> = (0..1_200)
            .map(|i| {
                if i < 1_000 {
                    a[i] as f64 + b[i] as f64
                } else {
                    0.0
                }
            })
            .collect();
        rep.check(
            "vec_add mismatched lens (1000,1500)->1200",
            &got,
            &want,
            abs_tol,
        );
    }

    // ------------------------------------------------------------ reductions
    // SAFETY: as above.
    let rd = unsafe { reduce::kernels::load(&ctx) }.expect("load reduce module");
    // n relative to TILE: exact multiple, one short, one over, tiny, huge.
    let rsizes: Vec<usize> = [
        1usize,
        TILE - 1,
        TILE,
        TILE + 1,
        7 * TILE + 13,
        1_048_576,
        1_048_576 + 999,
    ]
    .into_iter()
    .filter(|&n| !quick || n < 100_000)
    .collect();
    for &n in &rsizes {
        let data = rng.vec(n, 100.0);
        let d = DeviceBuffer::from_host(&stream, &data).unwrap();
        let blocks = n.div_ceil(TILE);
        for &block in &[32u32, 64, 128, 256] {
            let cfg = LaunchConfig1D::new(blocks as u32, block, 0);
            let mut o = DeviceBuffer::<f32>::zeroed(&stream, blocks).unwrap();
            // grid = ceil(n / TILE) blocks, one partial per block; out has `blocks` slots.
            let prep = rd.prepare_reduce_sum(cfg).expect("prepare reduce_sum");
            rd.reduce_sum(&stream, &prep, &d, &mut o)
                .expect("reduce_sum");
            let got = o.to_host_vec(&stream).unwrap();
            let want: Vec<f64> = (0..blocks)
                .map(|b| {
                    data[b * TILE..((b + 1) * TILE).min(n)]
                        .iter()
                        .map(|&x| x as f64)
                        .sum()
                })
                .collect();
            // f32 accumulation error of a window is bounded by eps * sum|x|
            // over that window (up to TILE terms), whatever the result's
            // magnitude — cancellation makes |sum| the wrong yardstick.
            let mags: Vec<f64> = (0..blocks)
                .map(|b| {
                    data[b * TILE..((b + 1) * TILE).min(n)]
                        .iter()
                        .map(|&x| (x as f64).abs())
                        .sum()
                })
                .collect();
            rep.check(
                &format!("reduce_sum partials b{block}"),
                &got,
                &want,
                |i, _| 1e-5 * mags[i].max(1.0),
            );
            let total_gpu: f64 = got.iter().map(|&x| x as f64).sum();
            let total_ref: f64 = data.iter().map(|&x| x as f64).sum();
            let total_mag: f64 = mags.iter().sum();
            rep.check(
                &format!("reduce_sum total b{block}"),
                &[total_gpu as f32],
                &[total_ref],
                |_, _| 1e-5 * total_mag.max(1.0),
            );

            let mut o = DeviceBuffer::<f32>::zeroed(&stream, blocks).unwrap();
            // As for reduce_sum.
            let prep = rd.prepare_reduce_max(cfg).expect("prepare reduce_max");
            rd.reduce_max(&stream, &prep, &d, &mut o)
                .expect("reduce_max");
            let got = o.to_host_vec(&stream).unwrap();
            let want: Vec<f64> = (0..blocks)
                .map(|b| {
                    data[b * TILE..((b + 1) * TILE).min(n)]
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64))
                })
                .collect();
            rep.check(
                &format!("reduce_max partials b{block}"),
                &got,
                &want,
                |_, _| 0.0,
            );
        }
    }
    // An all-negative window must not be polluted by the 0.0 padding of
    // reduce_max (identity is -inf) — n = TILE/2 + 3, all negative.
    {
        let n = TILE / 2 + 3;
        let data: Vec<f32> = rng
            .vec(n, 5.0)
            .into_iter()
            .map(|x| -x.abs() - 1.0)
            .collect();
        let d = DeviceBuffer::from_host(&stream, &data).unwrap();
        let mut o = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
        // One block, one partial.
        let prep = rd
            .prepare_reduce_max(LaunchConfig1D::new(1, 128, 0))
            .unwrap();
        rd.reduce_max(&stream, &prep, &d, &mut o).unwrap();
        let got = o.to_host_vec(&stream).unwrap();
        let want = [data.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64))];
        rep.check(
            "reduce_max all-negative padded window",
            &got,
            &want,
            |_, _| 0.0,
        );
    }

    // ---------------------------------------------------------------- matmul
    // SAFETY: as above.
    let mm = unsafe { matmul::kernels::load(&ctx) }.expect("load matmul module");
    {
        let a = rng.vec(M * K, 1.0);
        let b = rng.vec(K * N, 1.0);
        let a_d = DeviceBuffer::from_host(&stream, &a).unwrap();
        let b_d = DeviceBuffer::from_host(&stream, &b).unwrap();
        let want: Vec<f64> = (0..M * N)
            .map(|idx| {
                let (i, j) = (idx / N, idx % N);
                (0..K)
                    .map(|k| a[i * K + k] as f64 * b[k * N + j] as f64)
                    .sum()
            })
            .collect();
        let tiles = M.div_ceil(MT) * N.div_ceil(MT);
        // Persistent-block grids: one block for everything, fewer than tiles,
        // exactly tiles, more than tiles (idle blocks exit before any barrier).
        for &grid in &[1u32, 3, tiles as u32, 2 * tiles as u32] {
            let mut o = DeviceBuffer::<f32>::zeroed(&stream, M * N).unwrap();
            // Block must be MT*MT threads (the kernel flattens (r, c) from threadIdx); any 1-D grid.
            mm.matmul_tiled(
                &stream,
                &mm.prepare_matmul_tiled(LaunchConfig1D::new(grid, (MT * MT) as u32, 0))
                    .expect("prepare matmul"),
                &a_d,
                &b_d,
                &mut o,
            )
            .expect("matmul_tiled");
            let got = o.to_host_vec(&stream).unwrap();
            rep.check(
                &format!("matmul_tiled {M}x{K}x{N} MT={MT} grid={grid}"),
                &got,
                &want,
                |_, w| 1e-5 * w.abs().max(1.0) + 1e-5 * (K as f64).sqrt(),
            );
        }
        // Output shorter than M*N: the tail guard must hold (no write past out).
        let mut o = DeviceBuffer::<f32>::zeroed(&stream, M * N - 37).unwrap();
        // As above; the kernel bounds-checks every write against out.len().
        mm.matmul_tiled(
            &stream,
            &mm.prepare_matmul_tiled(LaunchConfig1D::new(tiles as u32, (MT * MT) as u32, 0))
                .expect("prepare matmul"),
            &a_d,
            &b_d,
            &mut o,
        )
        .expect("matmul_tiled short out");
        let got = o.to_host_vec(&stream).unwrap();
        rep.check(
            "matmul_tiled short out (M*N-37)",
            &got,
            &want[..M * N - 37],
            |_, w| 1e-5 * w.abs().max(1.0) + 1e-5 * (K as f64).sqrt(),
        );
    }

    stream.synchronize().expect("final sync");
    println!(
        "parity: {} cases, {} failures{}",
        rep.cases,
        rep.failures,
        if quick { " (quick mode)" } else { "" }
    );
    if rep.failures > 0 {
        std::process::exit(1);
    }
}
