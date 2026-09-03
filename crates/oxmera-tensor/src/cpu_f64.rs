//! The CPU backend's `f64` path (issue #27). Correct first, fast enough
//! second: every op is the plain reference formula over strided views, in
//! `f64` throughout, parallel over output chunks past the same threshold
//! the `f32` path uses. Research metrics — set-likelihood normalisers,
//! log-determinants, compensated sums — are what this serves; the `f32`
//! path keeps the tuned kernels.

use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{DType, Error, Result, Shape};
use rayon::prelude::*;

use crate::backend::{BinaryOp, ReduceOp, UnaryOp};
use crate::cpu_iter::OffsetWalker;
use crate::tensor::Tensor;

const PAR_THRESHOLD: usize = 16 * 1024;

pub(crate) fn f64_input<'t>(t: &'t Tensor, op: &'static str) -> Result<&'t [f64]> {
    if t.dtype() != DType::F64 {
        return Err(Error::UnsupportedDType {
            dtype: t.dtype(),
            op,
        });
    }
    t.storage().cpu()?.f64s()
}

impl UnaryOp {
    /// The `f64` reference semantics of the op.
    pub fn eval_f64(self, x: f64) -> f64 {
        match self {
            UnaryOp::Neg => -x,
            UnaryOp::Exp => x.exp(),
            UnaryOp::Ln => x.ln(),
            UnaryOp::Abs => x.abs(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Gelu => {
                const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
            }
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
}

impl BinaryOp {
    /// The `f64` reference semantics of the op.
    pub fn eval_f64(self, a: f64, b: f64) -> f64 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Pow => a.powf(b),
            BinaryOp::Maximum => a.max(b),
            BinaryOp::Minimum => a.min(b),
            BinaryOp::Gt => f64::from(a > b),
            BinaryOp::Eq => f64::from(a == b),
        }
    }
}

impl ReduceOp {
    /// The identity element, in `f64`.
    pub fn identity_f64(self) -> f64 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::Min => f64::INFINITY,
        }
    }

    /// Combine an accumulator with one element, in `f64`.
    pub fn combine_f64(self, acc: f64, x: f64) -> f64 {
        match self {
            ReduceOp::Sum => acc + x,
            ReduceOp::Max => acc.max(x),
            ReduceOp::Min => acc.min(x),
        }
    }
}

fn fill_par(out: &mut [f64], f: impl Fn(usize, &mut [f64]) + Sync) {
    if out.len() >= PAR_THRESHOLD {
        let chunk = PAR_THRESHOLD / 4;
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, c)| f(i * chunk, c));
    } else {
        f(0, out);
    }
}

pub(crate) fn unary(op: UnaryOp, a: &Tensor) -> Result<Tensor> {
    let src = f64_input(a, "unary")?;
    let mut out = vec![0.0f64; a.numel()];
    fill_par(&mut out, |start, chunk| {
        let mut w = OffsetWalker::at(a.layout(), start);
        for o in chunk.iter_mut() {
            *o = op.eval_f64(src[w.next_offset()]);
        }
    });
    Tensor::from_vec_f64(out, a.shape().clone())
}

pub(crate) fn binary(op: BinaryOp, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let out_shape = broadcast_shapes(a.shape(), b.shape())?;
    let av = a.broadcast_view(&out_shape)?;
    let bv = b.broadcast_view(&out_shape)?;
    let asrc = f64_input(&av, "binary")?;
    let bsrc = f64_input(&bv, "binary")?;
    let mut out = vec![0.0f64; out_shape.numel()];
    fill_par(&mut out, |start, chunk| {
        let mut wa = OffsetWalker::at(av.layout(), start);
        let mut wb = OffsetWalker::at(bv.layout(), start);
        for o in chunk.iter_mut() {
            *o = op.eval_f64(asrc[wa.next_offset()], bsrc[wb.next_offset()]);
        }
    });
    Tensor::from_vec_f64(out, out_shape)
}

pub(crate) fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let plan = crate::backend::plan_matmul(a.shape(), b.shape())?;
    let av = a.to_vec_f64()?;
    let bv = b.to_vec_f64()?;
    let (m, k, n) = (plan.m, plan.k, plan.n);
    let mut out = vec![0.0f64; plan.batch * m * n];
    out.par_chunks_mut(m * n).enumerate().for_each(|(bi, c)| {
        let ao = bi * plan.a_batch_stride;
        let bo = bi * plan.b_batch_stride;
        for i in 0..m {
            let row = &mut c[i * n..(i + 1) * n];
            for kk in 0..k {
                let x = av[ao + i * k + kk];
                if x == 0.0 {
                    continue;
                }
                let brow = &bv[bo + kk * n..bo + kk * n + n];
                for (o, &y) in row.iter_mut().zip(brow) {
                    *o += x * y;
                }
            }
        }
    });
    Tensor::from_vec_f64(out, plan.out_shape)
}

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

pub(crate) fn reduce(op: ReduceOp, a: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
    let src = f64_input(a, "reduce")?;
    let dims = a.dims().to_vec();
    let strides = a.layout().strides.values().to_vec();
    let base = a.layout().offset as isize;
    let (out_dims, kept) = split_axes(&dims, axes, keepdim);
    let reduced_dims: Vec<usize> = axes.iter().map(|&ax| dims[ax]).collect();
    let reduced_strides: Vec<isize> = axes.iter().map(|&ax| strides[ax]).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();
    let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
    let out_numel: usize = out_dims.iter().product();
    let empty = reduced_dims.contains(&0);
    let reduce_one = |out_i: usize| -> f64 {
        if empty {
            return op.identity_f64();
        }
        let mut rem = out_i;
        let mut offset = base;
        for (i, &d) in kept_dims.iter().enumerate().rev() {
            offset += (rem % d) as isize * kept_strides[i];
            rem /= d;
        }
        // Sums are Neumaier-compensated even in f64: a research metric that
        // reaches for f64 is usually one that cancels.
        let mut acc = op.identity_f64();
        let mut comp = 0.0f64;
        let mut coords = vec![0usize; reduced_dims.len()];
        let mut off = offset;
        loop {
            let x = src[off as usize];
            if op == ReduceOp::Sum {
                let t = acc + x;
                comp += if acc.abs() >= x.abs() {
                    (acc - t) + x
                } else {
                    (x - t) + acc
                };
                acc = t;
            } else {
                acc = op.combine_f64(acc, x);
            }
            let mut d = reduced_dims.len();
            loop {
                if d == 0 {
                    return if op == ReduceOp::Sum { acc + comp } else { acc };
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
    let out: Vec<f64> = if out_numel >= 1024 {
        (0..out_numel).into_par_iter().map(reduce_one).collect()
    } else {
        (0..out_numel).map(reduce_one).collect()
    };
    Tensor::from_vec_f64(out, Shape::new(out_dims))
}

pub(crate) fn argmax(a: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor> {
    let src = f64_input(a, "argmax")?;
    let dims = a.dims().to_vec();
    let strides = a.layout().strides.values().to_vec();
    let base = a.layout().offset as isize;
    let (out_dims, kept) = split_axes(&dims, &[dim], keepdim);
    let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
    let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();
    let (n, s) = (dims[dim], strides[dim]);
    if n == 0 {
        return Err(Error::InvalidArgument {
            op: "argmax",
            detail: format!("dimension {dim} has extent 0; argmax of nothing is undefined"),
        });
    }
    let out_numel: usize = out_dims.iter().product();
    let out: Vec<i64> = (0..out_numel)
        .map(|out_i| {
            let mut rem = out_i;
            let mut offset = base;
            for (i, &d) in kept_dims.iter().enumerate().rev() {
                offset += (rem % d) as isize * kept_strides[i];
                rem /= d;
            }
            let mut best = f64::NEG_INFINITY;
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

pub(crate) fn index_select(a: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
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
    // Logical order, then gather rows: correct for any layout.
    let src = a.to_vec_f64()?;
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();
    let mut out_dims = dims.clone();
    out_dims[dim] = idx.len();
    let mut out = Vec::with_capacity(outer * idx.len() * inner);
    for o in 0..outer {
        for &sel in &idx {
            let start = (o * dims[dim] + sel as usize) * inner;
            out.extend_from_slice(&src[start..start + inner]);
        }
    }
    Tensor::from_vec_f64(out, Shape::new(out_dims))
}

pub(crate) fn index_add(a: &Tensor, dim: usize, indices: &Tensor, srct: &Tensor) -> Result<Tensor> {
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
    let mut out = a.to_vec_f64()?;
    let s = srct.to_vec_f64()?;
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
            let dst = (o * dims[dim] + sel as usize) * inner;
            let src = (o * idx.len() + k) * inner;
            for j in 0..inner {
                out[dst + j] += s[src + j];
            }
        }
    }
    Tensor::from_vec_f64(out, a.shape().clone())
}

/// `cholesky` in f64 in and out (the reference routine is f64 internally
/// already; this keeps the result in f64 instead of rounding it).
pub(crate) fn cholesky(a: &Tensor) -> Result<Tensor> {
    let d = a.dims();
    let n = d[d.len() - 1];
    let batch: usize = d[..d.len() - 2].iter().product();
    let data = a.to_vec_f64()?;
    let as_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    // Reuse the reference for the pivot check, then redo the arithmetic in
    // f64 for the result: small matrices, so the double pass is cheap.
    crate::cpu_linalg::cholesky(&as_f32, batch, n)?;
    let mut out = vec![0.0f64; batch * n * n];
    for b in 0..batch {
        let m = &data[b * n * n..(b + 1) * n * n];
        let l = &mut out[b * n * n..(b + 1) * n * n];
        for j in 0..n {
            let mut dd = m[j * n + j];
            for k in 0..j {
                dd -= l[j * n + k] * l[j * n + k];
            }
            if dd.is_nan() || dd <= 0.0 || dd.is_infinite() {
                return Err(Error::InvalidArgument {
                    op: "cholesky",
                    detail: format!(
                        "matrix {b} is not positive definite (pivot {j} is {dd:e}); cholesky/logdet need an SPD input"
                    ),
                });
            }
            let ljj = dd.sqrt();
            l[j * n + j] = ljj;
            for i in j + 1..n {
                let mut s = m[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / ljj;
            }
        }
    }
    Tensor::from_vec_f64(out, a.shape().clone())
}
