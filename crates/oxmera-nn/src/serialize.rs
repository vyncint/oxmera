//! Model weights in the standard `safetensors` format.

use std::collections::HashMap;
use std::path::Path;

use oxmera_core::{Device, Error, Result, Shape};
use oxmera_tensor::tensor::Tensor;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::Module;

fn io_err(op: &'static str) -> impl Fn(std::io::Error) -> Error {
    move |e| Error::Io {
        op,
        detail: e.to_string(),
    }
}

/// Save a module's named parameters to `path` in safetensors format.
pub fn save(module: &dyn Module, path: impl AsRef<Path>) -> Result<()> {
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, param) in module.named_parameters("") {
        let value = param.value().to_device(Device::Cpu)?;
        let dims = value.dims().to_vec();
        let data = value.to_vec_f32()?;
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        buffers.push((name, dims, bytes));
    }
    let views: Vec<(String, TensorView<'_>)> = buffers
        .iter()
        .map(|(name, dims, bytes)| {
            TensorView::new(Dtype::F32, dims.clone(), bytes)
                .map(|v| (name.clone(), v))
                .map_err(|e| Error::Io {
                    op: "safetensors::save",
                    detail: e.to_string(),
                })
        })
        .collect::<Result<_>>()?;
    let serialized = safetensors::serialize(views, None).map_err(|e| Error::Io {
        op: "safetensors::save",
        detail: e.to_string(),
    })?;
    std::fs::write(path, serialized).map_err(io_err("safetensors::save"))
}

/// Load safetensors weights from `path` into a module's named parameters.
///
/// Every parameter must be present with a matching shape; extra tensors in
/// the file are ignored.
pub fn load(module: &dyn Module, path: impl AsRef<Path>) -> Result<()> {
    let bytes = std::fs::read(path).map_err(io_err("safetensors::load"))?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|e| Error::Io {
        op: "safetensors::load",
        detail: e.to_string(),
    })?;
    let by_name: HashMap<String, TensorView<'_>> = tensors.tensors().into_iter().collect();

    for (name, param) in module.named_parameters("") {
        let view = by_name.get(&name).ok_or_else(|| Error::Io {
            op: "safetensors::load",
            detail: format!("missing tensor {name}"),
        })?;
        if view.dtype() != Dtype::F32 {
            return Err(Error::Io {
                op: "safetensors::load",
                detail: format!("{name}: expected F32, got {:?}", view.dtype()),
            });
        }
        let dims = view.shape().to_vec();
        let expected = param.value().dims().to_vec();
        if dims != expected {
            return Err(Error::ShapeMismatch {
                expected: Shape::new(expected),
                got: Shape::new(dims),
                op: "safetensors::load",
            });
        }
        let (chunks, _rest) = view.data().as_chunks::<4>();
        let data: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
        param.set(Tensor::from_vec_f32(
            data,
            Shape::new(view.shape().to_vec()),
        )?);
    }
    Ok(())
}
