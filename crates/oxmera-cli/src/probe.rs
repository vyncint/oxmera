//! Probing the real machine. Only `doctor` without `--fixture` comes
//! here; goldens run on fixtures.

use std::process::Command;

use crate::report::{Devices, Host, Report, Toolchain};

fn first_line_of(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next().map(|l| l.trim().to_string())
}

fn sysctl(name: &str) -> Option<String> {
    first_line_of("sysctl", &["-n", name])
}

fn sysctl_u64(name: &str) -> Option<u64> {
    sysctl(name)?.parse().ok()
}

/// Probe the machine doctor is running on.
pub fn probe() -> Report {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let host = if os == "macos" {
        Host {
            chip: sysctl("machdep.cpu.brand_string"),
            p_cores: sysctl_u64("hw.perflevel0.physicalcpu").map(|v| v as u32),
            e_cores: sysctl_u64("hw.perflevel1.physicalcpu").map(|v| v as u32),
            memory_gb: sysctl_u64("hw.memsize").map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0)),
        }
    } else {
        Host {
            chip: None,
            p_cores: None,
            e_cores: None,
            memory_gb: None,
        }
    };

    #[cfg(target_os = "macos")]
    let (metal, metal_name, metal_budget_gb) = match oxmera::metal::device_summary() {
        Some((name, budget, _used)) => (
            true,
            Some(name),
            Some(budget as f64 / (1024.0 * 1024.0 * 1024.0)),
        ),
        None => (false, None, None),
    };
    #[cfg(not(target_os = "macos"))]
    let (metal, metal_name, metal_budget_gb) = (false, None, None);

    Report {
        os,
        arch,
        host,
        toolchain: Toolchain {
            rustc: first_line_of("rustc", &["--version"]),
            cargo: first_line_of("cargo", &["--version"]),
        },
        devices: Devices {
            cpu_threads: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            metal,
            metal_name,
            metal_budget_gb,
        },
    }
}
