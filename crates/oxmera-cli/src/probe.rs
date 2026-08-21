//! Probing the real machine. Only `doctor` without `--fixture` comes
//! here; tests never do — goldens run on fixtures so no golden depends on
//! the machine it was blessed on.

use std::path::Path;
use std::process::Command;

use crate::report::{Gpu, Report, Rung, Toolchain};

fn first_line_of(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next().map(|l| l.trim().to_string())
}

fn on_path(cmd: &str) -> bool {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path).any(|dir| dir.join(cmd).is_file())
}

/// The exercise manifest, if we are inside an oxmera checkout.
fn find_ladder() -> (bool, Vec<Rung>) {
    #[derive(serde::Deserialize)]
    struct Manifest {
        #[serde(rename = "exercise")]
        exercises: Vec<Rung>,
    }
    let mut dir = std::env::current_dir().ok();
    while let Some(d) = dir {
        let candidate = d.join("exercises/manifest.toml");
        if candidate.is_file() {
            if let Ok(text) = std::fs::read_to_string(&candidate) {
                if let Ok(m) = toml::from_str::<Manifest>(&text) {
                    return (true, m.exercises);
                }
            }
            return (false, Vec::new());
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    (false, Vec::new())
}

/// Probe the machine doctor is running on.
pub fn probe() -> Report {
    let (ladder_found, ladder) = find_ladder();
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    Report {
        gpu: Gpu {
            // Conservative: Metal is claimed only for Apple-Silicon macOS,
            // and only as a path that exists — not as a tested device.
            metal: os == "macos" && arch == "aarch64",
            cuda: on_path("nvidia-smi") || on_path("nvcc"),
        },
        toolchain: Toolchain {
            rustc: first_line_of("rustc", &["--version"]),
            cargo: first_line_of("cargo", &["--version"]),
            just: first_line_of("just", &["--version"]),
            reconverge: first_line_of("cargo-reconverge", &["--version"]),
            launchbound: first_line_of("launchbound", &["--version"]),
            container: on_path("container") || on_path("docker"),
        },
        os,
        arch,
        ladder,
        ladder_found,
    }
}
