//! The multi-threaded CPU backend implementation. Lives inside the
//! tensor crate so it can be registered lazily by `backend_for` — the CPU
//! reference is *always* available, with no link-order or life-before-main
//! caveats. The public face is the `oxmera-cpu` crate.

use std::sync::Arc;

use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{DType, Device, Error, Layout, Result, Shape, Strides};
use rayon::prelude::*;

use crate::backend::{Backend, BinaryOp, ReduceOp, UnaryOp};
use crate::cpu_iter::OffsetWalker;
use crate::cpu_matmul;
use crate::tensor::Tensor;

/// Below this element count, parallel dispatch costs more than it saves.
const PAR_THRESHOLD: usize = 16 * 1024;

/// The CPU backend. Stateless; one instance serves all CPU tensors.
#[derive(Debug, Default)]
pub struct CpuBackend;

/// Register the CPU backend explicitly. `backend_for` does this lazily,
/// so calling it is never required — only harmless.
pub fn register() {
    crate::backend::register_backend(Arc::new(CpuBackend));
}

fn f32_input<'t>(t: &'t Tensor, op: &'static str) -> Result<&'t [f32]> {
    if t.dtype() != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype: t.dtype(),
            op,
        });
    }
    t.storage().cpu()?.f32s()
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn unary(&self, op: UnaryOp, a: &Tensor) -> Result<Tensor> {
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::unary(op, a);
        }
        let src = f32_input(a, "unary")?;
        let numel = a.numel();
        let mut out = vec![0.0f32; numel];
        if a.layout().is_contiguous() {
            let start = a.layout().offset;
            let input = &src[start..start + numel];
            // Dispatch on the op once, outside the loop: each arm is its
            // own monomorphized, vectorizable loop. Leaving `op.eval` inside
            // the loop relies on LLVM hoisting the match, which it stopped
            // doing once this module grew (relu: 0.85 ms → 2.3 ms per 10M).
            match op {
                UnaryOp::Neg => map_contig(&mut out, input, |x| -x),
                UnaryOp::Exp => map_contig(&mut out, input, f32::exp),
                UnaryOp::Ln => map_contig(&mut out, input, f32::ln),
                UnaryOp::Abs => map_contig(&mut out, input, f32::abs),
                UnaryOp::Sqrt => map_contig(&mut out, input, f32::sqrt),
                UnaryOp::Sin => map_contig(&mut out, input, f32::sin),
                UnaryOp::Cos => map_contig(&mut out, input, f32::cos),
                UnaryOp::Tanh => map_contig(&mut out, input, f32::tanh),
                UnaryOp::Relu => map_contig(&mut out, input, |x| x.max(0.0)),
                UnaryOp::Gelu => map_contig(&mut out, input, |x| UnaryOp::Gelu.eval(x)),
                UnaryOp::Sigmoid => map_contig(&mut out, input, |x| UnaryOp::Sigmoid.eval(x)),
            }
        } else {
            walk_into(&mut out, a, numel, |x| op.eval(x));
        }
        Tensor::from_vec_f32(out, a.shape().clone())
    }

    fn binary(&self, op: BinaryOp, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                expected: a.dtype(),
                got: b.dtype(),
                op: "binary",
            });
        }
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::binary(op, a, b);
        }
        let out_shape = broadcast_shapes(a.shape(), b.shape())?;
        let av = a.broadcast_view(&out_shape)?;
        let bv = b.broadcast_view(&out_shape)?;
        let asrc = f32_input(&av, "binary")?;
        let bsrc = f32_input(&bv, "binary")?;
        let numel = out_shape.numel();
        let mut out = vec![0.0f32; numel];

        // Fast path: when both operands step through the innermost
        // dimension with stride 1 (contiguous) or 0 (broadcast — a row
        // vector, a column vector, a scalar), every output row is a tight
        // loop over two slices or a slice and a constant. Same-shape
        // arithmetic, `x - rowmax`, `x / rowsum`, `w * scalar` all land
        // here; only genuinely strided operands (transposed views) take
        // the odometer path below.
        if let Some(rows) = RowPlan::new(&out_shape, av.layout(), bv.layout()) {
            match op {
                BinaryOp::Add => rows.run(&mut out, asrc, bsrc, |x, y| x + y),
                BinaryOp::Sub => rows.run(&mut out, asrc, bsrc, |x, y| x - y),
                BinaryOp::Mul => rows.run(&mut out, asrc, bsrc, |x, y| x * y),
                BinaryOp::Div => rows.run(&mut out, asrc, bsrc, |x, y| x / y),
                BinaryOp::Pow => rows.run(&mut out, asrc, bsrc, |x, y| x.powf(y)),
                BinaryOp::Maximum => rows.run(&mut out, asrc, bsrc, |x, y| x.max(y)),
                BinaryOp::Minimum => rows.run(&mut out, asrc, bsrc, |x, y| x.min(y)),
                BinaryOp::Gt => rows.run(&mut out, asrc, bsrc, |x, y| f32::from(x > y)),
                BinaryOp::Eq => rows.run(&mut out, asrc, bsrc, |x, y| f32::from(x == y)),
            }
            return Tensor::from_vec_f32(out, out_shape);
        }

        let fill = |chunk_start: usize, chunk: &mut [f32]| {
            let mut wa = OffsetWalker::at(av.layout(), chunk_start);
            let mut wb = OffsetWalker::at(bv.layout(), chunk_start);
            for o in chunk.iter_mut() {
                *o = op.eval(asrc[wa.next_offset()], bsrc[wb.next_offset()]);
            }
        };

        if numel >= PAR_THRESHOLD {
            let chunk = PAR_THRESHOLD / 4;
            out.par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(i, c)| fill(i * chunk, c));
        } else {
            fill(0, &mut out);
        }
        Tensor::from_vec_f32(out, out_shape)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                expected: a.dtype(),
                got: b.dtype(),
                op: "matmul",
            });
        }
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::matmul(a, b);
        }
        cpu_matmul::matmul(a, b)
    }

    fn reduce(&self, op: ReduceOp, a: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::reduce(op, a, axes, keepdim);
        }
        let src = f32_input(a, "reduce")?;
        let dims = a.dims().to_vec();

        // Fast paths for contiguous data. `axes` arrives sorted and
        // deduplicated (normalize_axes). Three layouts cover almost every
        // reduction a model performs: everything (a loss), a trailing
        // block (softmax's max/sum over the last dim, row sums) and a
        // leading block (column sums, batch means).
        if a.layout().is_contiguous() && a.layout().offset == 0 && !dims.is_empty() {
            let numel = a.numel();
            let data = &src[..numel];
            let (out_dims, _) = split_axes(&dims, axes, keepdim);
            let out_shape = Shape::new(out_dims);
            let ndim = dims.len();
            let trailing = axes
                .iter()
                .enumerate()
                .all(|(i, &ax)| ax == ndim - axes.len() + i);
            let leading = axes.iter().enumerate().all(|(i, &ax)| ax == i);
            if axes.len() == ndim {
                let v = reduce_slice_par(op, data);
                return Tensor::from_vec_f32(vec![v], out_shape);
            }
            if trailing {
                let inner: usize = dims[ndim - axes.len()..].iter().product();
                let outer = out_shape.numel();
                let out: Vec<f32> = if inner == 0 {
                    vec![op.identity(); outer]
                } else if outer >= 64 || numel < PAR_THRESHOLD {
                    // Enough rows to parallelize across, or too small to matter.
                    let f = |i: usize| reduce_slice(op, &data[i * inner..(i + 1) * inner]);
                    if numel >= PAR_THRESHOLD {
                        (0..outer).into_par_iter().map(f).collect()
                    } else {
                        (0..outer).map(f).collect()
                    }
                } else {
                    // Few, long rows: parallelize inside each row instead.
                    (0..outer)
                        .map(|i| reduce_slice_par(op, &data[i * inner..(i + 1) * inner]))
                        .collect()
                };
                return Tensor::from_vec_f32(out, out_shape);
            }
            if leading {
                let inner = out_shape.numel();
                let outer: usize = dims[..axes.len()].iter().product();
                let mut out = vec![op.identity(); inner];
                if outer > 0 && inner > 0 {
                    // Walk the reduced rows in memory order, accumulating
                    // into a contiguous output slab — cache-friendly, and
                    // with the op dispatched once the inner loop
                    // vectorizes. Parallel over slab chunks.
                    match op {
                        ReduceOp::Sum => {
                            accumulate_rows(&mut out, data, outer, inner, |a, x| a + x)
                        }
                        ReduceOp::Max => accumulate_rows(&mut out, data, outer, inner, f32::max),
                        ReduceOp::Min => accumulate_rows(&mut out, data, outer, inner, f32::min),
                    }
                }
                return Tensor::from_vec_f32(out, out_shape);
            }
        }

        let strides = a.layout().strides.values().to_vec();
        let base = a.layout().offset;

        let (out_dims, kept): (Vec<usize>, Vec<usize>) = split_axes(&dims, axes, keepdim);
        let reduced_dims: Vec<usize> = axes.iter().map(|&ax| dims[ax]).collect();
        let reduced_strides: Vec<isize> = axes.iter().map(|&ax| strides[ax]).collect();
        let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
        let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();

        let out_numel: usize = out_dims.iter().product();
        // A reduced extent of 0 has nothing to combine: every output is
        // the identity (sum → 0, max → -inf, min → +inf). The odometer
        // below is do-while shaped and would read one element that does
        // not exist (issue #18).
        let empty_reduce = reduced_dims.contains(&0);
        let reduce_one = |out_i: usize| -> f32 {
            if empty_reduce {
                return op.identity();
            }
            // Decompose out_i over the kept dims to a base offset.
            let mut rem = out_i;
            let mut offset = base as isize;
            for (i, &d) in kept_dims.iter().enumerate().rev() {
                let coord = rem % d;
                rem /= d;
                offset += coord as isize * kept_strides[i];
            }
            // Walk the reduced subspace with an odometer.
            let mut acc = op.identity();
            let mut coords = vec![0usize; reduced_dims.len()];
            let mut off = offset;
            loop {
                acc = op.combine(acc, src[off as usize]);
                let mut d = reduced_dims.len();
                loop {
                    if d == 0 {
                        return acc;
                    }
                    d -= 1;
                    coords[d] += 1;
                    off += reduced_strides[d];
                    if coords[d] < reduced_dims[d] {
                        break;
                    }
                    off -= reduced_dims[d] as isize * reduced_strides[d];
                    coords[d] = 0;
                }
            }
        };

        let out: Vec<f32> = if out_numel >= 1024 {
            (0..out_numel).into_par_iter().map(reduce_one).collect()
        } else {
            (0..out_numel).map(reduce_one).collect()
        };
        Tensor::from_vec_f32(out, Shape::new(out_dims))
    }

    fn argmax(&self, a: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::argmax(a, dim, keepdim);
        }
        let src = f32_input(a, "argmax")?;
        let dims = a.dims().to_vec();
        let strides = a.layout().strides.values().to_vec();
        let base = a.layout().offset;
        let (out_dims, kept) = split_axes(&dims, &[dim], keepdim);
        let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
        let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();
        let (n, s) = (dims[dim], strides[dim]);
        if n == 0 {
            // There is no element to point at; fabricating index 0 would
            // send callers out of bounds downstream.
            return Err(Error::InvalidArgument {
                op: "argmax",
                detail: format!("dimension {dim} has extent 0; argmax of nothing is undefined"),
            });
        }

        let out_numel: usize = out_dims.iter().product();
        let out: Vec<i64> = (0..out_numel)
            .map(|out_i| {
                let mut rem = out_i;
                let mut offset = base as isize;
                for (i, &d) in kept_dims.iter().enumerate().rev() {
                    let coord = rem % d;
                    rem /= d;
                    offset += coord as isize * kept_strides[i];
                }
                let mut best = f32::NEG_INFINITY;
                let mut best_i = 0i64;
                for j in 0..n {
                    let v = src[(offset + j as isize * s) as usize];
                    if v > best {
                        best = v;
                        best_i = j as i64;
                    }
                }
                best_i
            })
            .collect();
        Tensor::from_vec_i64(out, Shape::new(out_dims))
    }

    fn contiguous(&self, a: &Tensor) -> Result<Tensor> {
        a.contiguous_data_crate()
    }

    fn download(&self, a: &Tensor) -> Result<Tensor> {
        a.contiguous_data_crate()
    }

    fn upload(&self, a: &Tensor) -> Result<Tensor> {
        a.contiguous_data_crate()
    }

    fn index_select(&self, a: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::index_select(a, dim, indices);
        }
        let src = f32_input(a, "index_select")?;
        let idx = indices.to_vec_i64()?;
        let dims = a.dims().to_vec();
        if dim >= dims.len() {
            return Err(Error::InvalidArgument {
                op: "index_select",
                detail: format!("dim {dim} out of range for rank {}", dims.len()),
            });
        }
        for &i in &idx {
            if i < 0 || i as usize >= dims[dim] {
                return Err(Error::IndexOutOfBounds {
                    index: vec![i.max(0) as usize],
                    shape: a.shape().clone(),
                });
            }
        }
        let mut out_dims = dims.clone();
        out_dims[dim] = idx.len();
        let out_shape = Shape::new(out_dims);
        let mut out = vec![0.0f32; out_shape.numel()];
        // Fill row by row along `dim` by narrowing per selected index.
        let mut walker_dims = out_shape.dims().to_vec();
        let _ = &mut walker_dims;
        let inner: usize = dims[dim + 1..].iter().product();
        let outer: usize = dims[..dim].iter().product();
        let strides = a.layout().strides.values();
        let base = a.layout().offset as isize;
        for o in 0..outer {
            // offset contribution of the outer coordinates
            let mut rem = o;
            let mut ooff = base;
            for d in (0..dim).rev() {
                let coord = rem % dims[d];
                rem /= dims[d];
                ooff += coord as isize * strides[d];
            }
            for (k, &sel) in idx.iter().enumerate() {
                let row = ooff + sel as isize * strides[dim];
                for j in 0..inner {
                    // inner coords decomposition
                    let mut rem2 = j;
                    let mut ioff = row;
                    for d in (dim + 1..dims.len()).rev() {
                        let coord = rem2 % dims[d];
                        rem2 /= dims[d];
                        ioff += coord as isize * strides[d];
                    }
                    out[(o * idx.len() + k) * inner + j] = src[ioff as usize];
                }
            }
        }
        Tensor::from_vec_f32(out, out_shape)
    }

    fn index_add(&self, a: &Tensor, dim: usize, indices: &Tensor, srct: &Tensor) -> Result<Tensor> {
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::index_add(a, dim, indices, srct);
        }
        let idx = indices.to_vec_i64()?;
        let dims = a.dims().to_vec();
        if dim >= dims.len() {
            return Err(Error::InvalidArgument {
                op: "index_add",
                detail: format!("dim {dim} out of range for rank {}", dims.len()),
            });
        }
        let mut expected = dims.clone();
        expected[dim] = idx.len();
        if srct.dims() != expected.as_slice() {
            return Err(Error::ShapeMismatch {
                expected: Shape::new(expected),
                got: srct.shape().clone(),
                op: "index_add",
            });
        }
        let mut out = a.to_vec_f32()?; // contiguous copy of a, logical order
        let s = srct.to_vec_f32()?;
        let inner: usize = dims[dim + 1..].iter().product();
        let outer: usize = dims[..dim].iter().product();
        for o in 0..outer {
            for (k, &sel) in idx.iter().enumerate() {
                if sel < 0 || sel as usize >= dims[dim] {
                    return Err(Error::IndexOutOfBounds {
                        index: vec![sel.max(0) as usize],
                        shape: a.shape().clone(),
                    });
                }
                let dst_row = (o * dims[dim] + sel as usize) * inner;
                let src_row = (o * idx.len() + k) * inner;
                for j in 0..inner {
                    out[dst_row + j] += s[src_row + j];
                }
            }
        }
        Tensor::from_vec_f32(out, a.shape().clone())
    }

    fn cholesky(&self, a: &Tensor) -> Result<Tensor> {
        if a.dtype() == DType::F64 {
            return crate::cpu_f64::cholesky(a);
        }
        f32_input(a, "cholesky")?;
        let d = a.dims();
        let n = d[d.len() - 1];
        let batch: usize = d[..d.len() - 2].iter().product();
        let data = a.to_vec_f32()?;
        let l = crate::cpu_linalg::cholesky(&data, batch, n)?;
        Tensor::from_vec_f32(l, a.shape().clone())
    }

    fn eigh(&self, a: &Tensor) -> Result<(Tensor, Tensor)> {
        let d = a.dims();
        let n = d[d.len() - 1];
        let batch: usize = d[..d.len() - 2].iter().product();
        let mut wshape = d[..d.len() - 1].to_vec();
        wshape[d.len() - 2] = n;
        // The Jacobi routine works in f64 internally; an f64 input keeps
        // its result in f64.
        if a.dtype() == DType::F64 {
            let data: Vec<f32> = a.to_vec_f64()?.iter().map(|&x| x as f32).collect();
            let (w, v) = crate::cpu_linalg::eigh(&data, batch, n);
            return Ok((
                Tensor::from_vec_f64(w.iter().map(|&x| x as f64).collect(), Shape::new(wshape))?,
                Tensor::from_vec_f64(v.iter().map(|&x| x as f64).collect(), a.shape().clone())?,
            ));
        }
        f32_input(a, "eigh")?;
        let data = a.to_vec_f32()?;
        let (w, v) = crate::cpu_linalg::eigh(&data, batch, n);
        Ok((
            Tensor::from_vec_f32(w, Shape::new(wshape))?,
            Tensor::from_vec_f32(v, a.shape().clone())?,
        ))
    }
}

/// Apply `f` element-by-element over a strided tensor into `out`.
///
/// Kept out of line: it is the cold branch of `unary`, and inlining its
/// row machinery into the caller measurably de-optimized the contiguous
/// fast path (relu over 10M elements went from 0.85 ms to 2.3 ms).
#[inline(never)]
fn walk_into(out: &mut [f32], a: &Tensor, numel: usize, f: impl Fn(f32) -> f32 + Sync) {
    let src = a
        .storage()
        .cpu()
        .expect("caller checked")
        .f32s()
        .expect("caller checked");
    // Rows whose innermost stride is 1 or 0 (a broadcast view being
    // materialized, a row-major slice) copy as tight loops.
    if let Some(rows) = RowPlan::new(a.shape(), a.layout(), a.layout()) {
        rows.run(out, src, src, |x, _| f(x));
        return;
    }
    if numel >= PAR_THRESHOLD {
        let chunk = PAR_THRESHOLD / 4;
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| {
            let mut w = OffsetWalker::at(a.layout(), i * chunk);
            for o in c.iter_mut() {
                *o = f(src[w.next_offset()]);
            }
        });
    } else {
        let mut w = OffsetWalker::at(a.layout(), 0);
        for o in out.iter_mut() {
            *o = f(src[w.next_offset()]);
        }
    }
}

/// Output dims after reducing `axes` (with or without keepdim), plus the
/// kept axis list.
fn split_axes(dims: &[usize], axes: &[usize], keepdim: bool) -> (Vec<usize>, Vec<usize>) {
    let mut out_dims = Vec::new();
    let mut kept = Vec::new();
    for (i, &d) in dims.iter().enumerate() {
        if axes.contains(&i) {
            if keepdim {
                out_dims.push(1);
            }
        } else {
            out_dims.push(d);
            kept.push(i);
        }
    }
    (out_dims, kept)
}

/// Reduce a contiguous slice serially.
///
/// Sums are Neumaier-compensated (Kahan–Babuška) across eight independent
/// lanes: the lanes let the loop vectorize (a plain fold is a serial chain
/// because f32 addition is not associative), and the compensation keeps
/// the result close to the exact sum even when large partials cancel —
/// a serial f32 fold of 100k values spanning ±50 that sum to −50 is off by
/// ~0.06; this is off by ~1e-4. Max/min fold directly.
fn reduce_slice(op: ReduceOp, data: &[f32]) -> f32 {
    match op {
        ReduceOp::Sum => {
            let mut sum = [0.0f32; 8];
            let mut comp = [0.0f32; 8];
            let (chunks, rest) = data.as_chunks::<8>();
            for c in chunks {
                for i in 0..8 {
                    let t = sum[i] + c[i];
                    // Neumaier: recover whichever operand lost low bits.
                    comp[i] += if sum[i].abs() >= c[i].abs() {
                        (sum[i] - t) + c[i]
                    } else {
                        (c[i] - t) + sum[i]
                    };
                    sum[i] = t;
                }
            }
            let mut total = 0.0f32;
            let mut ctotal = 0.0f32;
            for x in sum.iter().chain(comp.iter()).chain(rest.iter()) {
                let t = total + x;
                ctotal += if total.abs() >= x.abs() {
                    (total - t) + x
                } else {
                    (x - t) + total
                };
                total = t;
            }
            total + ctotal
        }
        _ => data
            .iter()
            .fold(op.identity(), |acc, &x| op.combine(acc, x)),
    }
}

/// Reduce a contiguous slice across threads when it is large enough:
/// per-chunk partials, then a serial combine (Sum/Max/Min are associative
/// up to rounding, and the chunking is fixed for a given length so a
/// result is deterministic run to run).
fn reduce_slice_par(op: ReduceOp, data: &[f32]) -> f32 {
    if data.len() < PAR_THRESHOLD {
        return reduce_slice(op, data);
    }
    let chunk = PAR_THRESHOLD / 4;
    let partials: Vec<f32> = data
        .par_chunks(chunk)
        .map(|c| reduce_slice(op, c))
        .collect();
    // Combining the partials is itself a reduction: reuse the compensated
    // sum so cancellation between large chunk totals is not lost.
    reduce_slice(op, &partials)
}

/// Row-wise execution plan for elementwise kernels: the output is walked
/// one innermost row at a time, and within a row each operand is either
/// a contiguous slice (inner stride 1) or a single broadcast value (inner
/// stride 0). Rows are handed to rayon in chunks.
struct RowPlan {
    inner: usize,
    rows: usize,
    a_outer: Layout,
    b_outer: Layout,
    a_step: usize,
    b_step: usize,
}

impl RowPlan {
    /// `None` when an operand is strided along the innermost dimension
    /// (a transposed view), or the shape is rank 0 / empty.
    fn new(shape: &Shape, a: &Layout, b: &Layout) -> Option<Self> {
        let dims = shape.dims();
        let ndim = dims.len();
        if ndim == 0 || dims.contains(&0) {
            return None;
        }
        let inner = dims[ndim - 1];
        let (sa, sb) = (a.strides.values()[ndim - 1], b.strides.values()[ndim - 1]);
        if !(sa == 0 || sa == 1) || !(sb == 0 || sb == 1) {
            return None;
        }
        let outer = |l: &Layout| Layout {
            shape: Shape::new(dims[..ndim - 1].to_vec()),
            strides: Strides::new(l.strides.values()[..ndim - 1].to_vec()),
            offset: l.offset,
        };
        Some(Self {
            inner,
            rows: shape.numel() / inner,
            a_outer: outer(a),
            b_outer: outer(b),
            a_step: sa as usize,
            b_step: sb as usize,
        })
    }

    #[inline(never)]
    fn run(&self, out: &mut [f32], a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32 + Sync) {
        let inner = self.inner;
        let fill_rows = |first_row: usize, chunk: &mut [f32]| {
            let mut wa = OffsetWalker::at(&self.a_outer, first_row);
            let mut wb = OffsetWalker::at(&self.b_outer, first_row);
            for row in chunk.chunks_mut(inner) {
                let (oa, ob) = (wa.next_offset(), wb.next_offset());
                match (self.a_step, self.b_step) {
                    (1, 1) => {
                        for (o, (&x, &y)) in row
                            .iter_mut()
                            .zip(a[oa..oa + inner].iter().zip(&b[ob..ob + inner]))
                        {
                            *o = f(x, y);
                        }
                    }
                    (1, _) => {
                        let y = b[ob];
                        for (o, &x) in row.iter_mut().zip(&a[oa..oa + inner]) {
                            *o = f(x, y);
                        }
                    }
                    (_, 1) => {
                        let x = a[oa];
                        for (o, &y) in row.iter_mut().zip(&b[ob..ob + inner]) {
                            *o = f(x, y);
                        }
                    }
                    _ => row.fill(f(a[oa], b[ob])),
                }
            }
        };
        let numel = self.rows * inner;
        if numel >= PAR_THRESHOLD && self.rows > 1 {
            let rows_per_chunk = (PAR_THRESHOLD / 4 / inner).max(1);
            out.par_chunks_mut(rows_per_chunk * inner)
                .enumerate()
                .for_each(|(i, c)| fill_rows(i * rows_per_chunk, c));
        } else {
            fill_rows(0, out);
        }
    }
}

/// `out[i] = f(input[i])` over contiguous data, parallel past the threshold.
#[inline]
fn map_contig(out: &mut [f32], input: &[f32], f: impl Fn(f32) -> f32 + Sync) {
    if out.len() >= PAR_THRESHOLD {
        out.par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(o, &x)| *o = f(x));
    } else {
        for (o, &x) in out.iter_mut().zip(input) {
            *o = f(x);
        }
    }
}

/// Column-style reduction: `out` is one contiguous slab of `inner`
/// elements and `data` is `outer` consecutive rows of it; every row is
/// folded into the slab. Parallel across slab chunks when large.
#[inline]
fn accumulate_rows(
    out: &mut [f32],
    data: &[f32],
    outer: usize,
    inner: usize,
    f: impl Fn(f32, f32) -> f32 + Sync,
) {
    let accumulate = |start: usize, chunk: &mut [f32]| {
        for r in 0..outer {
            let row = &data[r * inner + start..r * inner + start + chunk.len()];
            for (o, &x) in chunk.iter_mut().zip(row) {
                *o = f(*o, x);
            }
        }
    };
    if outer * inner >= PAR_THRESHOLD && inner >= 256 {
        let chunk = inner.div_ceil(rayon::current_num_threads()).max(256);
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, c)| accumulate(i * chunk, c));
    } else {
        accumulate(0, out);
    }
}
