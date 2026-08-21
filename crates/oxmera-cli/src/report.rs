//! The environment report: what doctor renders. One struct, two sources —
//! probed from the machine, or injected from a fixture for tests.

use serde::Deserialize;

/// Everything `oxmera doctor` knows about one machine.
#[derive(Debug, Deserialize)]
pub struct Report {
    /// Operating system family: "macos", "linux", or other.
    pub os: String,
    /// CPU architecture, e.g. "aarch64".
    pub arch: String,
    pub toolchain: Toolchain,
    pub gpu: Gpu,
    /// The exercise ladder, in order. Empty when no manifest was found.
    #[serde(default)]
    pub ladder: Vec<Rung>,
    /// Whether an exercises/manifest.toml was found and parsed.
    #[serde(default)]
    pub ladder_found: bool,
}

/// Tool presence and versions. `None` means not found on PATH.
#[derive(Debug, Deserialize)]
pub struct Toolchain {
    pub rustc: Option<String>,
    pub cargo: Option<String>,
    pub just: Option<String>,
    pub reconverge: Option<String>,
    pub launchbound: Option<String>,
    /// Apple `container` or another way to run the tier-1 image.
    pub container: bool,
}

/// GPU-shaped facts, probed conservatively — doctor never overclaims.
#[derive(Debug, Deserialize)]
pub struct Gpu {
    /// An Apple-Silicon Metal device is plausibly present (macOS/aarch64).
    pub metal: bool,
    /// A CUDA driver/toolkit is visible on this machine.
    pub cuda: bool,
}

/// One exercise rung, as the manifest records it.
#[derive(Debug, Deserialize)]
pub struct Rung {
    pub id: String,
    pub name: String,
    pub status: String,
}
