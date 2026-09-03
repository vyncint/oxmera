//! Small batched linear algebra on the last two dimensions: identity and
//! diagonal constructors, trace, Cholesky, log-determinant and the
//! symmetric eigen-decomposition.
//!
//! `eye`, `diag`, `diag_embed`, `trace`, `logdet` and `det` are composites
//! of recorded ops and therefore run and differentiate on every backend.
//! `cholesky` and `eigh` are backend primitives; a backend that declines
//! them falls back to the CPU implementation through a device round-trip,
//! the same contract `index_select` uses. Sizes are meant to be small
//! (covariance blocks, determinantal kernels), where the round-trip is
//! cheap and exactness matters more than throughput.

use oxmera_core::{DType, Device, Error, Result, Shape};

use crate::autograd::GradFn;
use crate::autograd::is_recording;
use crate::backend::{Backend, backend_for};
use crate::tensor::Tensor;

fn square_matrix_dims(t: &Tensor, op: &'static str) -> Result<(usize, usize)> {
    let d = t.dims();
    if d.len() < 2 || d[d.len() - 1] != d[d.len() - 2] {
        return Err(Error::InvalidArgument {
            op,
            detail: format!("needs a [.., n, n] tensor, got shape {:?}", t.shape()),
        });
    }
    let n = d[d.len() - 1];
    let batch: usize = d[..d.len() - 2].iter().product();
    Ok((batch, n))
}

/// Run a linear-algebra primitive on the tensor's backend, falling back
/// to a CPU round-trip when the backend declines.
fn dispatch_cpu_fallback<T>(
    t: &Tensor,
    f: impl Fn(&dyn Backend, &Tensor) -> Result<T>,
    upload: impl Fn(&dyn Backend, T) -> Result<T>,
) -> Result<T> {
    let backend = backend_for(t.device())?;
    match f(backend.as_ref(), t) {
        Err(Error::NotImplemented { .. }) if t.device() != Device::Cpu => {
            let cpu = backend.download(t)?;
            let out = f(backend_for(Device::Cpu)?.as_ref(), &cpu)?;
            upload(backend.as_ref(), out)
        }
        other => other,
    }
}

impl Tensor {
    /// The `n × n` identity matrix on the CPU.
    pub fn eye(n: usize) -> Tensor {
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        Tensor::from_vec_f32(v, Shape::from([n, n])).expect("lengths match by construction")
    }

    /// The `n × n` identity matrix on `device`.
    pub fn eye_on(n: usize, device: Device) -> Result<Tensor> {
        Tensor::eye(n).to_device(device)
    }

    /// The identity with the dtype and device of `like` (VJP plumbing).
    fn eye_like(n: usize, like: &Tensor) -> Result<Tensor> {
        Tensor::eye(n)
            .to_dtype(like.dtype())?
            .to_device(like.device())
    }

    /// The diagonal of every matrix in a `[.., n, n]` tensor, as `[.., n]`.
    /// Differentiable.
    pub fn diag(&self) -> Result<Tensor> {
        let (_, n) = square_matrix_dims(self, "diag")?;
        let eye = Tensor::eye_like(n, self)?;
        self.mul(&eye)?.sum(&[self.ndim() - 1])
    }

    /// Matrices with the vectors of a `[.., n]` tensor on their diagonals,
    /// as `[.., n, n]`. Differentiable.
    pub fn diag_embed(&self) -> Result<Tensor> {
        let d = self.dims();
        let Some(&n) = d.last() else {
            return Err(Error::InvalidArgument {
                op: "diag_embed",
                detail: "needs rank >= 1".into(),
            });
        };
        let eye = Tensor::eye_like(n, self)?;
        self.unsqueeze(self.ndim())?.mul(&eye)
    }

    /// The trace of every matrix in a `[.., n, n]` tensor, as `[..]`.
    /// Differentiable.
    pub fn trace(&self) -> Result<Tensor> {
        let diag = self.diag()?;
        diag.sum(&[diag.ndim() - 1])
    }

    /// Lower-triangular Cholesky factor `L` of every symmetric
    /// positive-definite matrix in a `[.., n, n]` tensor, `L Lᵀ = A`.
    ///
    /// Reads the lower triangle. A matrix that is not positive definite
    /// is a typed [`Error::InvalidArgument`] naming the batch index and
    /// pivot. Differentiable (Murray 2016); the backward pass runs on the
    /// host in `f64` and returns to the input's device.
    pub fn cholesky(&self) -> Result<Tensor> {
        square_matrix_dims(self, "cholesky")?;
        let out = dispatch_cpu_fallback(self, |be, t| be.cholesky(t), |be, l| be.upload(&l))?;
        if !(is_recording() && self.is_tracked()) {
            return Ok(out);
        }
        let l = out.clone();
        let shape = self.shape().clone();
        let device = self.device();
        Ok(out.with_grad_fn(GradFn {
            inputs: vec![self.clone()],
            vjp: Box::new(move |g: &Tensor| {
                let (batch, n) = square_matrix_dims(&l, "cholesky backward")?;
                let dtype = l.dtype();
                let lv = l
                    .to_device(Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_vec_f32()?;
                let gv = g
                    .to_device(Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_vec_f32()?;
                let grad = crate::cpu_linalg::cholesky_backward(&lv, &gv, batch, n);
                let grad = Tensor::from_vec_f32(grad, shape.clone())?
                    .to_dtype(dtype)?
                    .to_device(device)?;
                Ok(vec![Some(grad)])
            }),
        }))
    }

    /// `ln det A` of every SPD matrix in a `[.., n, n]` tensor, as `[..]`,
    /// through the Cholesky factor: `2 Σ ln diag(L)`. Differentiable
    /// (the gradient is `A⁻¹`, symmetrized).
    pub fn logdet(&self) -> Result<Tensor> {
        let l = self.cholesky()?;
        let d = l.diag()?;
        d.ln()?.sum(&[d.ndim() - 1])?.mul_scalar(2.0)
    }

    /// `det A` of every SPD matrix in a `[.., n, n]` tensor, as `[..]`,
    /// through the Cholesky factor. Differentiable. For an indefinite
    /// matrix use `eigh` — this is the SPD determinant.
    pub fn det(&self) -> Result<Tensor> {
        self.logdet()?.exp()
    }

    /// Eigen-decomposition of every symmetric matrix in a `[.., n, n]`
    /// tensor: eigenvalues ascending as `[.., n]` and orthonormal
    /// eigenvectors as the columns of `[.., n, n]` (`A V = V Λ`). The
    /// full matrix is read and symmetrized. Not differentiable.
    pub fn eigh(&self) -> Result<(Tensor, Tensor)> {
        square_matrix_dims(self, "eigh")?;
        dispatch_cpu_fallback(
            self,
            |be, t| be.eigh(t),
            |be, (w, v)| Ok((be.upload(&w)?, be.upload(&v)?)),
        )
    }
}
