//! The oxmera terminal surface.
//!
//! `oxmera doctor` reports what this machine can and cannot do: toolchain,
//! the three cost tiers, backend paths, and the exercise ladder. It is
//! pure infrastructure — the one part of oxmera that is allowed to work
//! before the exercises are solved.
//!
//! Determinism contract (termlens goldens depend on it): given a fixture,
//! the output is byte-identical across runs — no clocks, no durations, no
//! absolute paths, no animation.

mod doctor;
mod probe;
mod report;

use std::process::ExitCode;

const USAGE: &str = "usage: oxmera <doctor [--fixture <path>] | --version>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("oxmera {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("doctor") => match doctor::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("oxmera doctor: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}
