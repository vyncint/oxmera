//! The training dashboard through a real PTY, in deterministic replay
//! mode: golden frames at two geometries, keypress handling, live resize,
//! and the 100-iteration stress. Replay frames contain no clocks,
//! durations, or absolute paths.
//!
//! **Sync policy: `snapshot_after`, never a sleep.** It waits for a fact
//! and then for the picture to hold still, returning that settled instant,
//! so a repaint cannot land between the wait and the read. Teardown lives
//! in `signals.rs`, which asks the question this file cannot: was the
//! terminal given back?
//!
//! Regenerate goldens after an intentional UI change with
//! `OXMERA_BLESS=1 cargo test -p oxmera-cli --test tui`.

use std::path::{Path, PathBuf};
use std::{env, fs};

use termlens::{Key, Screen, Terminal};

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

/// Spawn the replay dashboard and hand back it and its settled screen.
///
/// The completion hint is the last thing drawn, so it is the right fact to
/// wait on; `snapshot_after` then waits for the picture to stop moving, so
/// the `Screen` returned is a whole frame rather than a possibly
/// half-painted one.
fn spawn(size: (u16, u16)) -> termlens::Result<(Terminal, Screen)> {
    let mut t = termlens::bin!(
        "oxmera",
        size(size.0, size.1),
        args(["train", "--replay"]),
        arg(fixture())
    )?;
    let screen = t.snapshot_after(|s| s.contains("training complete — press q"))?;
    Ok((t, screen))
}

/// Quit, and check the terminal came back.
///
/// Every test here goes through this, so every test is also a teardown
/// test — an application that stopped restoring on `q` would fail all of
/// them rather than none of them, which is what happened before 0.4.0 for
/// the signal paths (#40).
fn quit(mut t: Terminal, context: &str) -> termlens::Result<()> {
    t.send(Key::Char('q'))?;
    let status = t.wait_exit()?;
    assert!(status.success(), "{context}: exited with {status:?}");
    assert!(
        !t.screen().alternate_screen(),
        "{context}: left the shell inside the alternate screen"
    );
    Ok(())
}

#[test]
fn dashboard_at_100x45() -> termlens::Result<()> {
    let (t, screen) = spawn((100, 45))?;
    assert!(screen.contains("oxmera train"), "header: {screen}");
    assert!(screen.contains("device metal"), "device label: {screen}");
    assert!(screen.contains("0.2005"), "final loss: {screen}");
    assert!(screen.contains("95.3%"), "final accuracy: {screen}");
    assert!(screen.contains("5790 samples/s"), "throughput: {screen}");
    assert!(screen.contains("1.2 / 27.0 GB unified"), "memory: {screen}");
    assert!(screen.contains("6/6"), "epoch gauge: {screen}");
    assert!(
        screen.alternate_screen(),
        "the dashboard should be holding the alternate screen while it runs"
    );
    assert_golden("tui-replay-100x45.txt", &screen.to_string(), "dashboard");
    quit(t, "dashboard")
}

/// Live resize: the same session re-laid-out at a new geometry.
///
/// `wait_stable` rather than `wait_idle`: a resize makes the application
/// repaint, and what this test needs is *the repaint finished*, not *the
/// bytes stopped*. Waiting on bytes can settle between two writes of one
/// frame.
#[test]
fn resize_relayouts_the_frame() -> termlens::Result<()> {
    let (mut t, _) = spawn((100, 45))?;
    t.resize(80, 30)?;
    let screen = t.wait_stable(std::time::Duration::from_millis(150))?;
    assert_golden("tui-replay-80x30.txt", &screen.to_string(), "resized");
    quit(t, "resized")
}

#[test]
fn esc_also_quits() -> termlens::Result<()> {
    let (mut t, _) = spawn((100, 45))?;
    t.send(Key::Esc)?;
    let status = t.wait_exit()?;
    assert!(status.success(), "esc exited with {status:?}");
    assert!(
        !t.screen().alternate_screen(),
        "esc left the shell inside the alternate screen"
    );
    Ok(())
}

/// The 100-iteration stress: a full replay must paint the identical final
/// frame every time — no races, no mid-paint captures, no
/// nondeterminism.
#[test]
fn stress_100_iterations_are_identical() -> termlens::Result<()> {
    let (first, first_screen) = spawn((100, 45))?;
    let reference = normalize(&first_screen.to_string());
    quit(first, "stress reference")?;
    for i in 1..100 {
        let (t, screen) = spawn((100, 45))?;
        let frame = normalize(&screen.to_string());
        assert_eq!(reference, frame, "iteration {i} painted a different frame");
        quit(t, "stress")?;
    }
    Ok(())
}
