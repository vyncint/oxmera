//! Serialization failure modes: a failed load must not mutate the module,
//! and a duplicate parameter name must be a typed error rather than a panic
//! from inside safetensors.

use oxmera_core::Result;
use oxmera_nn::{Module, Param, serialize};
use oxmera_tensor::tensor::Tensor;

#[derive(Debug)]
struct Two {
    a: Param,
    b: Param,
}

impl Module for Two {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Param)> {
        vec![("a".into(), self.a.clone()), ("b".into(), self.b.clone())]
    }
}

#[derive(Debug)]
struct Dup {
    p: Param,
}

impl Module for Dup {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
    fn parameters(&self) -> Vec<Param> {
        vec![self.p.clone(), self.p.clone()]
    }
    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Param)> {
        vec![("w".into(), self.p.clone()), ("w".into(), self.p.clone())]
    }
}

fn leaf(values: &[f32]) -> Param {
    Param::new(Tensor::from_slice(values, [values.len()]).unwrap())
}

#[test]
fn a_failed_load_leaves_every_parameter_untouched() {
    let src = Two {
        a: leaf(&[9.0, 9.0]),
        b: leaf(&[9.0, 9.0]),
    };
    let path = std::env::temp_dir().join("oxmera_load_atomic.safetensors");
    serialize::save(&src, &path).unwrap();

    // "a" matches the checkpoint; "b" does not, so the load fails on "b"
    // only after "a" would already have been written.
    let victim = Two {
        a: leaf(&[1.0, 1.0]),
        b: leaf(&[1.0, 1.0, 1.0]),
    };
    assert!(serialize::load(&victim, &path).is_err());
    assert_eq!(
        victim.a.value().to_vec_f32().unwrap(),
        vec![1.0, 1.0],
        "a failed load must not half-apply the checkpoint"
    );
    assert_eq!(victim.b.value().to_vec_f32().unwrap(), vec![1.0, 1.0, 1.0]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn duplicate_parameter_names_are_a_typed_error() {
    let d = Dup {
        p: leaf(&[1.0, 2.0]),
    };
    let path = std::env::temp_dir().join("oxmera_dup_names.safetensors");
    let e = serialize::save(&d, &path).unwrap_err();
    assert!(e.to_string().contains("duplicate parameter name"), "{e}");
    let _ = std::fs::remove_file(&path);
}
