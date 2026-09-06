//! Reverse-mode autograd for oxmera.
//!
//! The tape lives with the tensor (`oxmera_tensor::autograd`) so every op
//! can record onto it; this crate re-exports the user surface and adds
//! **finite-difference gradient checking**, the tool that keeps every VJP
//! honest.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use oxmera_tensor::autograd::{NoGradGuard, is_recording, no_grad};
pub use oxmera_tensor::tensor::Tensor;

use oxmera_core::{Error, Result, Shape};

/// Compare analytic gradients against central finite differences.
///
/// `f` maps the inputs to a scalar tensor. Every input is treated as a
/// gradient-accumulating leaf; for each element the numeric derivative
/// `(f(x+eps) - f(x-eps)) / 2eps` must match the analytic one within
/// `tol` (absolute or relative, whichever is looser).
pub fn gradcheck(
    f: impl Fn(&[Tensor]) -> Result<Tensor>,
    inputs: &[Tensor],
    eps: f32,
    tol: f32,
) -> Result<()> {
    // Analytic pass.
    let leaves: Vec<Tensor> = inputs
        .iter()
        .map(|t| t.detach().requires_grad_(true))
        .collect();
    let out = f(&leaves)?;
    if out.numel() != 1 {
        return Err(Error::InvalidArgument {
            op: "gradcheck",
            detail: format!("f must return a scalar, got shape {:?}", out.shape()),
        });
    }
    out.backward()?;
    let analytic: Vec<Option<Vec<f32>>> = leaves
        .iter()
        .map(|l| l.grad().map(|g| g.to_vec_f32()).transpose())
        .collect::<Result<_>>()?;

    // Numeric pass, element by element.
    for (i, input) in inputs.iter().enumerate() {
        let base = input.to_vec_f32()?;
        let shape = input.shape().clone();
        let Some(agrad) = &analytic[i] else {
            return Err(Error::InvalidArgument {
                op: "gradcheck",
                detail: format!("input {i} received no analytic gradient"),
            });
        };
        for (j, &a) in agrad.iter().enumerate() {
            let numeric = central_difference(&f, inputs, i, j, &base, &shape, eps)?;
            let denom = 1.0f32.max(a.abs()).max(numeric.abs());
            if (a - numeric).abs() / denom > tol {
                return Err(Error::InvalidArgument {
                    op: "gradcheck",
                    detail: format!(
                        "input {i} element {j}: analytic {a} vs numeric {numeric} (tol {tol})"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn central_difference(
    f: &impl Fn(&[Tensor]) -> Result<Tensor>,
    inputs: &[Tensor],
    which: usize,
    elem: usize,
    base: &[f32],
    shape: &Shape,
    eps: f32,
) -> Result<f32> {
    let eval = |delta: f32| -> Result<f32> {
        let mut data = base.to_vec();
        data[elem] += delta;
        let mut probe: Vec<Tensor> = inputs.to_vec();
        probe[which] =
            Tensor::from_vec_f32(data, shape.clone())?.to_device(inputs[which].device())?;
        let out = no_grad(|| f(&probe))?;
        out.to_device(oxmera_core::Device::Cpu)?
            .to_vec_f32()
            .map(|v| v[0])
    };
    Ok((eval(eps)? - eval(-eps)?) / (2.0 * eps))
}

/// What this crate can do, for `oxmera doctor`. See
/// [`oxmera_tensor::CAPABILITIES`] for why the list lives beside the code.
pub const CAPABILITIES: &[(&str, &str)] = &[(
    "autograd",
    "reverse-mode tape, finite-difference verified, no_grad guard",
)];
