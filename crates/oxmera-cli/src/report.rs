//! The environment report `oxmera doctor` renders. One struct, two
//! sources — probed from the machine, or injected from a fixture so the
//! termlens goldens never depend on the machine they were blessed on.

use serde::Deserialize;

/// Everything doctor knows about one machine.
#[derive(Debug, Deserialize)]
pub struct Report {
    /// Operating system family: "macos", "linux", or other.
    pub os: String,
    /// CPU architecture, e.g. "aarch64".
    pub arch: String,
    /// Physical host details.
    pub host: Host,
    /// Compiler toolchain versions.
    pub toolchain: Toolchain,
    /// Compute devices oxmera can use here.
    pub devices: Devices,
}

/// Hardware identity.
#[derive(Debug, Deserialize)]
pub struct Host {
    /// Chip/SoC name (e.g. "Apple M3 Pro").
    pub chip: Option<String>,
    /// Performance-core count, when the platform distinguishes.
    pub p_cores: Option<u32>,
    /// Efficiency-core count.
    pub e_cores: Option<u32>,
    /// Total physical memory, GB.
    pub memory_gb: Option<f64>,
}

/// Tool presence and versions. `None` means not found on PATH.
#[derive(Debug, Deserialize)]
pub struct Toolchain {
    /// `rustc --version`.
    pub rustc: Option<String>,
    /// `cargo --version`.
    pub cargo: Option<String>,
}

/// Compute devices.
#[derive(Debug, Deserialize)]
pub struct Devices {
    /// Threads the CPU backend parallelizes across.
    pub cpu_threads: u32,
    /// Whether a Metal device is registered.
    pub metal: bool,
    /// Metal device name, when present.
    pub metal_name: Option<String>,
    /// Metal recommended working-set budget, GB.
    pub metal_budget_gb: Option<f64>,
}
