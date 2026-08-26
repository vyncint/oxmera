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

# The nightly research workspace: format and check only — no execution,
# no GPU, no CUDA toolkit required. rustup installs the pinned nightly
# from research/rust-toolchain.toml on first use.
research:
    cd research/oxmera-cuda && cargo fmt --all --check
    cd research/oxmera-cuda && cargo check --all-targets

