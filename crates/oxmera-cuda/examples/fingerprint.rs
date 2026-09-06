//! Print the fingerprint of the CUDA source, for `just ptx`.
//!
//! An example rather than a script so that the fingerprint the recipe
//! writes is produced by the *same function* the test compares against.
//! A shell one-liner reimplementing FNV-1a beside a Rust one is two
//! implementations of one constant-sensitive algorithm, and the first
//! version of that function already got the prime wrong by a digit.

fn main() {
    let source = include_str!("../kernels.cu");
    println!("{}", oxmera_cuda::source_fingerprint(source));
}
