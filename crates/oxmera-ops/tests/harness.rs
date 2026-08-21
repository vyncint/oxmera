//! Harness placeholder for the ops seam.
//!
//! `oxmera-ops` is signatures only, permanently — there is nothing to run
//! here, and there never will be. The behavioural specs for the operations
//! live in `exercises/` (rung A4 and onward) and run against backends.
//! This test asserts the one thing this crate owns: the traits stay
//! object-safe, because the runtime seam dispatches through `dyn`.

use oxmera_ops::{BinaryOps, MatmulOps, ReduceOps, UnaryOps};

#[test]
#[ignore = "compile-time assertion only; unignore never — replaced by backend specs in exercises/"]
fn traits_are_object_safe() {
    fn _takes_dyn(_: &dyn UnaryOps, _: &dyn BinaryOps, _: &dyn MatmulOps, _: &dyn ReduceOps) {}
}
