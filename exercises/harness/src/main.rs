//! The exercise-ladder harness.
//!
//! Reads `exercises/manifest.toml` and drives each rung according to its
//! status: a `todo` rung is *compiled* (so an unclimbed ladder never shows
//! red), a `solved` rung is *verified* (its specs run, and kernel rungs
//! re-pass the convergence gate). This is infrastructure — it contains no
//! exercise solutions and never will.
//!
//! Usage: `exercise-harness <list | run <id> | run-all>` — or, from the
//! repo root, `just exercise <id>` / `just exercises`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use serde::{Deserialize, Serialize};

/// The manifest: one entry per rung, in ladder order.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Manifest {
    #[serde(rename = "exercise")]
    exercises: Vec<Exercise>,
}

/// One rung of the ladder.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Exercise {
    id: String,
    name: String,
    tier: Tier,
    /// What the rung costs to run ("$0" for tiers A–C; tier D is metered).
    cost: String,
    /// The tools that gate the rung.
    tools: Vec<String>,
    status: Status,
    /// Directory relative to the repo root. Absent while `planned`.
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    kind: Kind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
enum Tier {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum Status {
    /// Built, unsolved: the harness compiles it and nothing more.
    Todo,
    /// Solved by the maintainer: the harness runs its full verification.
    Solved,
    /// Not built yet; listed so the ladder shape is visible.
    Planned,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum Kind {
    /// A spec crate: `cargo test` (or `--no-run` while todo).
    Crate,
    /// A cuda-oxide kernel crate: `cargo check`, plus the reconverge gate
    /// once solved.
    Kernel,
    /// A written exercise: solved means a non-empty ANSWER.md exists.
    Written,
}

fn repo_root() -> PathBuf {
    // harness lives at <root>/exercises/harness
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("harness sits two levels below the repo root")
        .to_path_buf()
}

fn load_manifest(root: &Path) -> Result<Manifest, String> {
    let path = root.join("exercises/manifest.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))
}

fn run_step(dir: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    let pretty = format!("{program} {}", args.join(" "));
    println!("    $ {pretty}  (in {})", dir.display());
    // The harness itself runs under `cargo run`, and rustup's shim exports
    // RUSTUP_TOOLCHAIN (plus CARGO/RUSTC paths) into child processes —
    // which would silently override each exercise's own
    // rust-toolchain.toml. Scrub them so per-directory resolution wins.
    let status = Command::new(program)
        .args(args)
        .current_dir(dir)
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env_remove("RUSTDOC")
        .status()
        .map_err(|e| format!("failed to spawn `{pretty}`: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{pretty}` failed with {status}"))
    }
}

fn run_exercise(root: &Path, ex: &Exercise) -> Result<(), String> {
    println!(
        "==> {} — {} [tier {:?}, {}] status: {:?}",
        ex.id, ex.name, ex.tier, ex.cost, ex.status
    );
    let Some(rel) = &ex.path else {
        println!("    planned, not built yet — nothing to run");
        return Ok(());
    };
    let dir = root.join(rel);
    match (ex.kind, ex.status) {
        (_, Status::Planned) => {
            println!("    planned — nothing to run");
            Ok(())
        }
        (Kind::Crate, Status::Todo) => run_step(&dir, "cargo", &["test", "--no-run"]),
        (Kind::Crate, Status::Solved) => run_step(&dir, "cargo", &["test"]),
        (Kind::Kernel, Status::Todo) => run_step(&dir, "cargo", &["check"]),
        (Kind::Kernel, Status::Solved) => {
            run_step(&dir, "cargo", &["check"])?;
            run_step(&dir, "cargo", &["reconverge", "check", "--strict"])
        }
        (Kind::Written, Status::Todo) => {
            println!("    written exercise — solved when ANSWER.md is written");
            Ok(())
        }
        (Kind::Written, Status::Solved) => {
            let answer = dir.join("ANSWER.md");
            let text = std::fs::read_to_string(&answer)
                .map_err(|e| format!("solved written exercise needs {}: {e}", answer.display()))?;
            if text.trim().len() < 80 {
                return Err(format!(
                    "{} is too short to be an explanation ({} bytes)",
                    answer.display(),
                    text.trim().len()
                ));
            }
            println!("    ANSWER.md present ({} bytes)", text.trim().len());
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();
    let manifest = match load_manifest(&root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mode = args.first().map(String::as_str).unwrap_or("list");
    match mode {
        "list" => {
            for ex in &manifest.exercises {
                println!(
                    "{:4} {:?}  {:8}  {} — {}",
                    ex.id,
                    ex.tier,
                    format!("{:?}", ex.status).to_lowercase(),
                    ex.cost,
                    ex.name
                );
            }
            ExitCode::SUCCESS
        }
        "run" => {
            let Some(id) = args.get(1) else {
                eprintln!("usage: exercise-harness run <id>");
                return ExitCode::FAILURE;
            };
            let Some(ex) = manifest.exercises.iter().find(|e| &e.id == id) else {
                eprintln!("error: no exercise with id `{id}`");
                return ExitCode::FAILURE;
            };
            match run_exercise(&root, ex) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        "run-all" => {
            let mut failed = Vec::new();
            for ex in &manifest.exercises {
                if let Err(e) = run_exercise(&root, ex) {
                    eprintln!("error: {e}");
                    failed.push(ex.id.clone());
                }
            }
            if failed.is_empty() {
                println!("ladder: every built rung is green for its status");
                ExitCode::SUCCESS
            } else {
                eprintln!("ladder: failed rungs: {}", failed.join(", "));
                ExitCode::FAILURE
            }
        }
        other => {
            eprintln!("usage: exercise-harness <list | run <id> | run-all>  (got `{other}`)");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips() {
        let root = repo_root();
        let text = std::fs::read_to_string(root.join("exercises/manifest.toml")).unwrap();
        let parsed: Manifest = toml::from_str(&text).unwrap();
        let re_serialized = toml::to_string(&parsed).unwrap();
        let re_parsed: Manifest = toml::from_str(&re_serialized).unwrap();
        assert_eq!(parsed, re_parsed);
        assert!(!parsed.exercises.is_empty());
    }

    #[derive(Debug, Deserialize)]
    struct ExerciseFile {
        exercise: ExerciseMeta,
    }

    #[derive(Debug, Deserialize)]
    struct ExerciseMeta {
        id: String,
        name: String,
        tier: Tier,
        cost: String,
        tools: Vec<String>,
        status: Status,
    }

    /// The anti-rot guard: a rung whose exercise.toml drifts from the
    /// manifest is worse than no rung.
    #[test]
    fn per_exercise_toml_agrees_with_the_manifest() {
        let root = repo_root();
        let text = std::fs::read_to_string(root.join("exercises/manifest.toml")).unwrap();
        let parsed: Manifest = toml::from_str(&text).unwrap();
        for ex in parsed
            .exercises
            .iter()
            .filter(|e| e.status != Status::Planned)
        {
            let rel = ex.path.as_ref().unwrap();
            let file = root.join(rel).join("exercise.toml");
            let meta: ExerciseFile =
                toml::from_str(&std::fs::read_to_string(&file).unwrap_or_else(|e| {
                    panic!("{}: every built rung carries an exercise.toml: {e}", ex.id)
                }))
                .unwrap_or_else(|e| panic!("{}: {e}", file.display()));
            let m = meta.exercise;
            assert_eq!(m.id, ex.id, "{}", file.display());
            assert_eq!(m.name, ex.name, "{}", file.display());
            assert_eq!(m.tier, ex.tier, "{}", file.display());
            assert_eq!(m.cost, ex.cost, "{}", file.display());
            assert_eq!(m.tools, ex.tools, "{}", file.display());
            assert_eq!(
                m.status,
                ex.status,
                "{}: status must match the manifest",
                file.display()
            );
        }
    }

    #[test]
    fn every_built_rung_has_a_real_path() {
        let root = repo_root();
        let text = std::fs::read_to_string(root.join("exercises/manifest.toml")).unwrap();
        let parsed: Manifest = toml::from_str(&text).unwrap();
        for ex in &parsed.exercises {
            match ex.status {
                Status::Planned => {
                    assert!(ex.path.is_none(), "{}: planned rungs have no path", ex.id)
                }
                _ => {
                    let rel = ex
                        .path
                        .as_ref()
                        .unwrap_or_else(|| panic!("{}: built rung needs a path", ex.id));
                    assert!(
                        root.join(rel).is_dir(),
                        "{}: path {rel} does not exist",
                        ex.id
                    );
                }
            }
        }
    }
}
