//! `oxmera doctor` through a real PTY on hermetic fixtures — one golden
//! per environment shape plus the 100-iteration stress. No frame contains
//! a clock, a duration, or an absolute path.
//!
//! **Sync policy: `snapshot_after`, never a sleep.** It waits for a fact
//! about the screen and then for the picture to hold still, and hands back
//! that settled instant — so there is no gap between "the thing appeared"
//! and "the screen was read" for a repaint to land in. The older shape
//! here was `wait_until(pred)` then `wait_idle(quiet)` then `screen()`,
//! which is three instants and waits on *bytes* stopping rather than on
//! the picture stopping.
//!
//! Regenerate goldens after an intentional output change with
//! `OXMERA_BLESS=1 cargo test -p oxmera-cli --test doctor`.

use std::path::{Path, PathBuf};
use std::{env, fs};

use termlens::Screen;

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

/// Run doctor on a fixture and hand back the settled screen.
///
/// `bin!` supplies `size`/`env_clear`/`timeout`/the binary path, and turns
/// a misspelled binary name into a compile error rather than a spawn
/// failure at run time.
///
/// The exit is checked here rather than left to `Drop`: `doctor` is a
/// program that prints and exits, so "did it exit 0" is half of what the
/// test is for. The alternate screen is checked too — `doctor` should
/// never take it, and a report that quietly started doing so would
/// otherwise look identical in the golden.
fn doctor_screen(fixture_name: &str) -> termlens::Result<Screen> {
    let mut t = termlens::bin!(
        "oxmera",
        size(100, 45),
        args(["doctor", "--fixture"]),
        arg(fixture(fixture_name))
    )?;
    let screen = t.snapshot_after(|s| s.contains("doctor: report complete"))?;
    let status = t.wait_exit()?;
    assert!(status.success(), "doctor exited with {status:?}");
    assert!(
        !screen.alternate_screen(),
        "doctor is a plain report and must never take the alternate screen"
    );
    Ok(screen)
}

#[test]
fn no_gpu_shape() -> termlens::Result<()> {
    let screen = doctor_screen("no-gpu.toml")?;
    // One predicate per instant, and all of it about the same screen —
    // this is a captured `Screen`, not a live one, so the two reads cannot
    // race each other.
    assert!(
        screen.contains("metal   not available on this host"),
        "{screen}"
    );
    assert!(
        screen.contains("cuda    not available on this host"),
        "{screen}"
    );
    assert_golden("doctor-no-gpu-100x45.txt", &screen.to_string(), "no-gpu");
    Ok(())
}

#[test]
fn metal_shape() -> termlens::Result<()> {
    let screen = doctor_screen("metal.toml")?;
    assert!(screen.contains("Apple M3 Pro (fixture)"), "{screen}");
    assert!(screen.contains("unified memory budget 27.0 GB"), "{screen}");
    assert!(screen.contains("5 performance + 6 efficiency"), "{screen}");
    assert_golden("doctor-metal-100x45.txt", &screen.to_string(), "metal");
    Ok(())
}

#[test]
fn cuda_shape() -> termlens::Result<()> {
    let screen = doctor_screen("cuda.toml")?;
    assert!(screen.contains("NVIDIA A10G (fixture)"), "{screen}");
    assert!(screen.contains("compute capability 8.6"), "{screen}");
    assert!(screen.contains("metal   not available"), "{screen}");
    assert_golden("doctor-cuda-100x45.txt", &screen.to_string(), "cuda");
    Ok(())
}

/// The 100-iteration stress: the same fixture must paint the same frame
/// every single time.
#[test]
fn stress_100_iterations_are_identical() -> termlens::Result<()> {
    let first = normalize(&doctor_screen("metal.toml")?.to_string());
    for i in 1..100 {
        let frame = normalize(&doctor_screen("metal.toml")?.to_string());
        assert_eq!(first, frame, "iteration {i} painted a different frame");
    }
    Ok(())
}
