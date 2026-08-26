//! The training dashboard through a real PTY, in deterministic replay
//! mode: golden frames at two geometries, keypress handling, live resize,
//! and the 100-iteration stress. Sync policy: wait_until on rendered
//! content, then wait_idle; never sleep. Replay frames contain no clocks,
//! durations, or absolute paths.
//!
//! Regenerate goldens after an intentional UI change with
//! `OXMERA_BLESS=1 cargo test -p oxmera-cli --test tui`.

use std::path::{Path, PathBuf};
use std::time::Duration;
use std::{env, fs};

use termlens::{Key, Terminal};

const QUIET: Duration = Duration::from_millis(150);
const TIMEOUT: Duration = Duration::from_secs(10);

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/train-replay.toml")
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

/// Spawn the replay dashboard and sync on the completed frame. A quiet
/// PTY is not a painted PTY: the completion hint is the last thing drawn.
fn spawn(size: (u16, u16)) -> Terminal {
    let mut t = Terminal::builder()
        .size(size.0, size.1)
        .env_clear()
        .timeout(TIMEOUT)
        .arg("train")
        .arg("--replay")
        .arg(fixture())
        .spawn(env!("CARGO_BIN_EXE_oxmera"))
        .expect("failed to spawn the dashboard in a PTY");
    t.wait_until(|s| s.to_string().contains("training complete — press q"))
        .expect("replay never completed");
    t.wait_idle(QUIET).expect("wait_idle");
    t
}

fn quit(mut t: Terminal, context: &str) {
    t.send(Key::Char('q')).expect("send q");
    let status = t.wait_exit().expect("dashboard did not exit after q");
    assert!(status.success(), "{context}: exited with {status:?}");
}

#[test]
fn dashboard_at_100x45() {
    let t = spawn((100, 45));
    let screen = t.screen().to_string();
    assert!(screen.contains("oxmera train"), "header");
    assert!(screen.contains("device metal"), "device label");
    assert!(screen.contains("0.2005"), "final loss visible");
    assert!(screen.contains("95.3%"), "final accuracy visible");
    assert!(screen.contains("5790 samples/s"), "throughput visible");
    assert!(screen.contains("1.2 / 27.0 GB unified"), "memory visible");
    assert!(screen.contains("6/6"), "epoch gauge complete");
    assert_golden("tui-replay-100x45.txt", &screen, "dashboard");
    quit(t, "dashboard");
}

/// Live resize: the same session re-laid-out at a new geometry.
#[test]
fn resize_relayouts_the_frame() {
    let mut t = spawn((100, 45));
    t.resize(80, 30).expect("resize");
    t.wait_idle(QUIET).expect("post-resize idle");
    assert_golden("tui-replay-80x30.txt", &t.screen().to_string(), "resized");
    quit(t, "resized");
}

#[test]
fn esc_also_quits() {
    let mut t = spawn((100, 45));
    t.send(Key::Esc).expect("send esc");
    let status = t.wait_exit().expect("no exit after esc");
    assert!(status.success());
}

/// The 100-iteration stress: a full replay must paint the identical final
/// frame every time — no races, no mid-paint captures, no
/// nondeterminism.
#[test]
fn stress_100_iterations_are_identical() {
    let first = spawn((100, 45));
    let reference = normalize(&first.screen().to_string());
    quit(first, "stress reference");
    for i in 1..100 {
        let t = spawn((100, 45));
        let frame = normalize(&t.screen().to_string());
        assert_eq!(reference, frame, "iteration {i} painted a different frame");
        quit(t, "stress");
    }
}
