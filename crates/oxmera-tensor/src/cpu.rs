//! The multi-threaded CPU backend implementation. Lives inside the
//! tensor crate so it can be registered lazily by `backend_for` — the CPU
//! reference is *always* available, with no link-order or life-before-main
//! caveats. The public face is the `oxmera-cpu` crate.

use std::sync::Arc;

use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{DType, Device, Error, Result, Shape};
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
        let src = f32_input(a, "unary")?;
        let numel = a.numel();
        let mut out = vec![0.0f32; numel];
        if a.layout().is_contiguous() {
            let start = a.layout().offset;
            let input = &src[start..start + numel];
            if numel >= PAR_THRESHOLD {
                out.par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(o, &x)| *o = op.eval(x));
            } else {
                for (o, &x) in out.iter_mut().zip(input) {
                    *o = op.eval(x);
                }
            }
        } else {
            walk_into(&mut out, a, numel, |x| op.eval(x));
        }
        Tensor::from_vec_f32(out, a.shape().clone())
    }

    fn binary(&self, op: BinaryOp, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_shapes(a.shape(), b.shape())?;
        let av = a.broadcast_view(&out_shape)?;
        let bv = b.broadcast_view(&out_shape)?;
        let asrc = f32_input(&av, "binary")?;
        let bsrc = f32_input(&bv, "binary")?;
        let numel = out_shape.numel();
        let mut out = vec![0.0f32; numel];

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
        cpu_matmul::matmul(a, b)
    }

    fn reduce(&self, op: ReduceOp, a: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        let src = f32_input(a, "reduce")?;
        let dims = a.dims().to_vec();
        let strides = a.layout().strides.values().to_vec();
        let base = a.layout().offset;

        let (out_dims, kept): (Vec<usize>, Vec<usize>) = split_axes(&dims, axes, keepdim);
        let reduced_dims: Vec<usize> = axes.iter().map(|&ax| dims[ax]).collect();
        let reduced_strides: Vec<isize> = axes.iter().map(|&ax| strides[ax]).collect();
        let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
        let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();

        let out_numel: usize = out_dims.iter().product();
        let reduce_one = |out_i: usize| -> f32 {
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
        let src = f32_input(a, "argmax")?;
        let dims = a.dims().to_vec();
        let strides = a.layout().strides.values().to_vec();
        let base = a.layout().offset;
        let (out_dims, kept) = split_axes(&dims, &[dim], keepdim);
        let kept_strides: Vec<isize> = kept.iter().map(|&ax| strides[ax]).collect();
        let kept_dims: Vec<usize> = kept.iter().map(|&ax| dims[ax]).collect();
        let (n, s) = (dims[dim], strides[dim]);

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
}

/// Apply `f` element-by-element over a strided tensor into `out`.
fn walk_into(out: &mut [f32], a: &Tensor, numel: usize, f: impl Fn(f32) -> f32 + Sync) {
    let src = a
        .storage()
        .cpu()
        .expect("caller checked")
        .f32s()
        .expect("caller checked");
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
