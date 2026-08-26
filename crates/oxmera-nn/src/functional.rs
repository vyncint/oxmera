//! Composite tensor functions layers share: concatenation, padding, and
//! one-hot encoding. Everything here is built from differentiable
//! primitives, so gradients flow without bespoke VJPs.

use oxmera_core::{Error, Result, Shape};
use oxmera_tensor::tensor::Tensor;

/// Concatenate tensors along `dim`. Differentiable (built on
/// `index_add`, whose gradient is `index_select`).
pub fn cat(parts: &[Tensor], dim: usize) -> Result<Tensor> {
    let Some(first) = parts.first() else {
        return Err(Error::InvalidArgument {
            op: "cat",
            detail: "no tensors given".into(),
        });
    };
    let mut out_dims = first.dims().to_vec();
    if dim >= out_dims.len() {
        return Err(Error::InvalidArgument {
            op: "cat",
            detail: format!("dim {dim} out of range for rank {}", out_dims.len()),
        });
    }
    out_dims[dim] = parts.iter().map(|t| t.dims()[dim]).sum();
    let mut out = Tensor::zeros(Shape::new(out_dims)).to_device(first.device())?;
    let mut cursor = 0usize;
    for part in parts {
        let len = part.dims()[dim];
        let indices: Vec<i64> = (cursor..cursor + len).map(|i| i as i64).collect();
        let indices = Tensor::from_vec_i64(indices, Shape::from([len]))?;
        out = out.index_add(dim, &indices, part)?;
        cursor += len;
    }
    Ok(out)
}

/// Zero-pad `dim` by `before`/`after` elements. Differentiable.
pub fn pad_dim(t: &Tensor, dim: usize, before: usize, after: usize) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(t.clone());
    }
    let mut dims = t.dims().to_vec();
    let len = dims[dim];
    dims[dim] = len + before + after;
    let out = Tensor::zeros(Shape::new(dims)).to_device(t.device())?;
    let indices: Vec<i64> = (before..before + len).map(|i| i as i64).collect();
    let indices = Tensor::from_vec_i64(indices, Shape::from([len]))?;
    out.index_add(dim, &indices, t)
}

/// One-hot encode `targets` (`I64`, shape `[n]`) into `[n, classes]` on
/// the CPU. A constant — gradients never flow into targets.
pub fn one_hot(targets: &Tensor, classes: usize) -> Result<Tensor> {
    let idx = targets.to_vec_i64()?;
    let n = idx.len();
    let mut data = vec![0.0f32; n * classes];
    for (row, &c) in idx.iter().enumerate() {
        if c < 0 || c as usize >= classes {
            return Err(Error::IndexOutOfBounds {
                index: vec![c.max(0) as usize],
                shape: Shape::from([classes]),
            });
        }
        data[row * classes + c as usize] = 1.0;
    }
    Tensor::from_vec_f32(data, Shape::from([n, classes]))
}
