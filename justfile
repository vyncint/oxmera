# oxmera local loop.
#
# Gate on the exit code, never on output: `just ci && git commit -sS ...`.
# Piping through `tail` (or anything else) swallows the exit code.

default:
    @just --list

# The full local gate. Green here before any push, on every commit.
ci: fmt clippy check test doc deny research

# Docs must build warning-free; O1's gate made this permanent.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

fmt:
    cargo fmt --all --check

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

check:
    cargo check --workspace --all-targets

test:
    cargo test --workspace

deny:
    cargo deny check

# Regenerate the shipped CUDA PTX and its source fingerprint, together.
#
# Needs `nvcc` on PATH; this is the one recipe that does. `kernels.ptx` and
# `kernels.ptx.source` MUST be written by the same command, because the
# whole point of the fingerprint is that the two files were produced from
# one source at one moment — see the test
# `the_shipped_ptx_was_built_from_the_shipped_source`. Editing the kernels
# and running anything less than this recipe is the failure it exists to
# catch.
#
# compute_75 (Turing) is the floor: it is what `needs_cc` claims and what
# every driver from the 12.8 series can load. A newer -arch would silently
# drop older cards.
ptx:
    cd crates/oxmera-cuda && nvcc -arch=compute_75 -O3 -ptx kernels.cu -o kernels.ptx
    cd crates/oxmera-cuda && cargo run --quiet --example fingerprint > kernels.ptx.source
    @echo "regenerated kernels.ptx and kernels.ptx.source — commit both"

# The nightly research workspace: format and check only — no execution,
# no GPU, no CUDA toolkit required. rustup installs the pinned nightly
# from research/rust-toolchain.toml on first use.
research:
    cd research/oxmera-cuda-oxide && cargo fmt --all --check
    cd research/oxmera-cuda-oxide && cargo check --all-targets
    cd research/oxmera-cuda-oxide && cargo test --release

