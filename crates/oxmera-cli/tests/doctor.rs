//! O5 gate tests: `oxmera doctor` through a real PTY on hermetic
//! fixtures — one golden per environment shape (no GPU / Metal / CUDA)
//! plus the 100-iteration stress. Sync policy: wait_until on rendered
//! content ("doctor: report complete" is the last line and appears in no
//! static text before it), then wait_idle; never sleep. No frame contains
//! a clock, a duration, or an absolute path.
//!
//! Regenerate goldens after an intentional output change with
//! `OXMERA_BLESS=1 cargo test -p oxmera-cli --test doctor`.

use std::path::{Path, PathBuf};
use std::time::Duration;
use std::{env, fs};

use termlens::Terminal;

const QUIET: Duration = Duration::from_millis(150);
const TIMEOUT: Duration = Duration::from_secs(10);

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn normalize(frame: &str) -> String {
    let joined = frame
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    joined.trim_end().to_string()
}

fn assert_golden(name: &str, screen: &str, context: &str) {
    let path = golden_path(name);
    let actual = normalize(screen);
    if env::var_os("OXMERA_BLESS").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, format!("{actual}\n")).unwrap();
    }
    let expected = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing golden {name}; bless with OXMERA_BLESS=1"));
    assert_eq!(
        normalize(&expected),
        actual,
        "{context}: frame differs from golden {name}\n--- rendered ---\n{screen}"
    );
}

/// Run doctor on a fixture and hand back the settled screen. A quiet PTY
/// is not a painted PTY: sync on the report's final line first.
fn doctor_screen(fixture_name: &str) -> String {
    let mut t = Terminal::builder()
        .size(100, 45)
        .env_clear()
        .timeout(TIMEOUT)
        .arg("doctor")
        .arg("--fixture")
        .arg(fixture(fixture_name))
        .spawn(env!("CARGO_BIN_EXE_oxmera"))
        .expect("failed to spawn oxmera doctor in a PTY");
    t.wait_until(|s| s.to_string().contains("doctor: report complete"))
        .expect("report never completed");
    t.wait_idle(QUIET).expect("wait_idle");
    let screen = t.screen().to_string();
    let status = t.wait_exit().expect("doctor did not exit");
    assert!(status.success(), "doctor exited with {status:?}");
    screen
}

#[test]
fn no_gpu_shape() {
    let screen = doctor_screen("no-gpu.toml");
    assert!(
        screen.contains("install cargo-reconverge"),
        "missing-tool hint visible"
    );
    assert!(
        screen.contains("metered"),
        "tier 2 must always read as metered"
    );
    assert_golden("doctor-no-gpu-100x45.txt", &screen, "no-gpu");
}

#[test]
fn metal_shape() {
    let screen = doctor_screen("metal.toml");
    assert!(
        screen.contains("no convergence gate exists on this path"),
        "the Metal banner is not optional"
    );
    assert!(screen.contains("1 solved"), "ladder counts visible");
    assert_golden("doctor-metal-100x45.txt", &screen, "metal");
}

#[test]
fn cuda_shape() {
    let screen = doctor_screen("cuda.toml");
    assert!(
        screen.contains("metered — approved GPU sessions only"),
        "a visible toolkit must not read as permission to spend"
    );
    assert!(screen.contains("per-part"), "the per-part warning stays");
    assert_golden("doctor-cuda-100x45.txt", &screen, "cuda");
}

/// The 100-iteration stress: the same fixture must paint the same frame
/// every single time. Catches nondeterminism (ordering, env leakage,
/// mid-paint captures) the single-shot goldens can miss.
#[test]
fn stress_100_iterations_are_identical() {
    let first = normalize(&doctor_screen("no-gpu.toml"));
    for i in 1..100 {
        let frame = normalize(&doctor_screen("no-gpu.toml"));
        assert_eq!(first, frame, "iteration {i} painted a different frame");
    }
}
