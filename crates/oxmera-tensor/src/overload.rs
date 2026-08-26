//! `std::ops` sugar for tensors.
//!
//! Operators must return a value, not a `Result`, so these impls panic on
//! shape/device/dtype errors. Use the named methods (`Tensor::add`, …)
//! when the failure should be handled instead of being a bug.

use crate::tensor::Tensor;

macro_rules! binary_operator {
    ($trait:ident, $method:ident, $tensor_method:ident) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$tensor_method(self, rhs)
                    .unwrap_or_else(|e| panic!(concat!("tensor ", stringify!($method), ": {}"), e))
            }
        }
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                std::ops::$trait::$method(&self, &rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                std::ops::$trait::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                std::ops::$trait::$method(self, &rhs)
            }
        }
        impl std::ops::$trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                let rhs =
                    Tensor::scalar_on(self, rhs).unwrap_or_else(|e| panic!("scalar operand: {e}"));
                std::ops::$trait::$method(self, &rhs)
            }
        }
        impl std::ops::$trait<f32> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                std::ops::$trait::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for f32 {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                let lhs =
                    Tensor::scalar_on(rhs, self).unwrap_or_else(|e| panic!("scalar operand: {e}"));
                std::ops::$trait::$method(&lhs, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for f32 {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                std::ops::$trait::$method(self, &rhs)
            }
        }
    };
}

binary_operator!(Add, add, add);
binary_operator!(Sub, sub, sub);
binary_operator!(Mul, mul, mul);
binary_operator!(Div, div, div);

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self).unwrap_or_else(|e| panic!("tensor neg: {e}"))
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        std::ops::Neg::neg(&self)
    }
}
