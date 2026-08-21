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
    cd research && cargo fmt --all --check
    cd research && cargo check --workspace --all-targets

# Run one exercise rung by id (a1..b4): compiles it while todo, verifies
# it once solved.
exercise id:
    cargo run -q -p exercise-harness -- run {{id}}

# Run the whole ladder. On a fresh clone every rung is todo and this
# exits 0 — an unclimbed ladder is never red, a climbed rung can never
# silently break. B-tier rungs pull the pinned nightly on first use.
exercises:
    cargo run -q -p exercise-harness -- run-all
