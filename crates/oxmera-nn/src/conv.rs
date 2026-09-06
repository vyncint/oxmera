//! 2-D convolution via im2col + batched matmul — built entirely from
//! differentiable primitives (`index_select`, `cat`, `matmul`), so the
//! backward pass comes from the tape, not a bespoke kernel.

use oxmera_core::{Error, Result, Shape};
use oxmera_tensor::tensor::Tensor;

use crate::functional::{cat, pad_dim};
use crate::param::Param;
use crate::{Module, init};

/// 2-D convolution: input `[n, c_in, h, w]`, output
/// `[n, c_out, h_out, w_out]`.
#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: Param,
    bias: Option<Param>,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: usize,
    padding: usize,
}

impl Conv2d {
    /// A convolution with Kaiming-uniform weights and zero bias.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Self {
        let fan_in = in_channels * kernel.0 * kernel.1;
        Self {
            weight: Param::new(init::kaiming_uniform(
                [out_channels, in_channels, kernel.0, kernel.1],
                fan_in,
                seed,
            )),
            bias: Some(Param::new(init::zeros([out_channels]))),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
        }
    }

    /// The `[c_out, c_in, kh, kw]` weight handle.
    pub fn weight(&self) -> &Param {
        &self.weight
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::check_param_dtype("Conv2d", input, &self.parameters())?;
        let dims = input.dims();
        if dims.len() != 4 || dims[1] != self.in_channels {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([0, self.in_channels, 0, 0]),
                got: input.shape().clone(),
                op: "Conv2d",
            });
        }
        let (n, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let (kh, kw) = self.kernel;
        let (s, p) = (self.stride, self.padding);
        let h_out = (h + 2 * p - kh) / s + 1;
        let w_out = (w + 2 * p - kw) / s + 1;

        let x = pad_dim(&pad_dim(input, 2, p, p)?, 3, p, p)?;

        // im2col: one slab per kernel offset, each [n, c, h_out, w_out],
        // concatenated along the channel dim in (kh, kw)-major order.
        let mut slabs = Vec::with_capacity(kh * kw);
        for oh in 0..kh {
            let rows: Vec<i64> = (0..h_out).map(|i| (oh + i * s) as i64).collect();
            let rows = Tensor::from_vec_i64(rows, Shape::from([h_out]))?;
            let xr = x.index_select(2, &rows)?;
            for ow in 0..kw {
                let cols: Vec<i64> = (0..w_out).map(|i| (ow + i * s) as i64).collect();
                let cols = Tensor::from_vec_i64(cols, Shape::from([w_out]))?;
                slabs.push(xr.index_select(3, &cols)?);
            }
        }
        let patches = cat(&slabs, 1)?; // [n, kh*kw*c, h_out, w_out]
        let patches =
            patches.reshape(Shape::from([n, kh * kw * self.in_channels, h_out * w_out]))?;

        // Weight reordered to match the (kh, kw)-major patch layout:
        // [c_out, c_in, kh, kw] -> [c_out, kh, kw, c_in] -> [c_out, khkw*c].
        let wt = self
            .weight
            .value()
            .to_device(input.device())?
            .permute(&[0, 2, 3, 1])?
            .contiguous()?
            .reshape(Shape::from([self.out_channels, kh * kw * self.in_channels]))?;
        let wt = wt
            .unsqueeze(0)?
            .broadcast_to(Shape::from([
                n,
                self.out_channels,
                kh * kw * self.in_channels,
            ]))?
            .contiguous()?;

        let mut y = wt.matmul(&patches)?; // [n, c_out, h_out*w_out]
        if let Some(bias) = &self.bias {
            let b = bias.value().to_device(input.device())?;
            y = y.add(&b.reshape(Shape::from([1, self.out_channels, 1]))?)?;
        }
        y.reshape(Shape::from([n, self.out_channels, h_out, w_out]))
    }

    fn parameters(&self) -> Vec<Param> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        let mut named = vec![(format!("{prefix}weight"), self.weight.clone())];
        if let Some(b) = &self.bias {
            named.push((format!("{prefix}bias"), b.clone()));
        }
        named
    }
}
