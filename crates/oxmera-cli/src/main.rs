//! The oxmera terminal surface.
//!
//! - `oxmera doctor` — what this machine can do: hardware, devices,
//!   framework capabilities. Deterministic given a fixture.
//! - `oxmera train` — train a demo MLP on a synthetic dataset, on CPU or
//!   Metal, with a plain reporter or the live `--tui` dashboard;
//!   `--replay` renders a recorded run for deterministic PTY testing.

mod doctor;
mod probe;
mod report;
mod restore;
mod train;
mod tui;

use std::process::ExitCode;

const USAGE: &str = "usage: oxmera <doctor [--fixture <path>] | train [--device cpu|metal|cuda] [--epochs N] [--tui] [--replay <path>] | --version | --help>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("oxmera {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        // An explicit request for help is a success on stdout; only an
        // unknown or missing subcommand is usage-on-stderr + exit 1.
        Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("doctor") => report_outcome("doctor", doctor::run(&args[1..])),
        Some("train") => report_outcome("train", train::run(&args[1..])),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn report_outcome(what: &str, outcome: Result<(), String>) -> ExitCode {
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("oxmera {what}: {e}");
            ExitCode::FAILURE
        }
    }
}
