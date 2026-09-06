//! The two exits nobody writes code for: a signal, and the ordinary quit
//! that must keep working after the signal handling is installed.
//!
//! A TUI borrows the terminal. Every one of these tests asks the same
//! question — *was it given back?* — through a real pty, because that is
//! the only place the answer exists: `alternate_screen()` is a property of
//! the emulator's state, not of anything the process printed.
//!
//! Unix only: `signal` has no meaning on Windows and `restore::on_signal`
//! is a documented no-op there.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use termlens::{Key, Signal, Terminal};

const TIMEOUT: Duration = Duration::from_secs(10);

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/train-replay.toml")
}

/// The dashboard, painted and settled, holding the terminal.
fn dashboard() -> termlens::Result<Terminal> {
    let mut t = Terminal::builder()
        .size(100, 45)
        .env_clear()
        .timeout(TIMEOUT)
        .arg("train")
        .arg("--replay")
        .arg(fixture())
        .spawn(env!("CARGO_BIN_EXE_oxmera"))?;
    // `snapshot_after` waits for the fact *and then* for the picture to
    // hold still, so what comes back is a settled screen rather than a
    // possibly half-painted one.
    let s = t.snapshot_after(|s| s.contains("training complete — press q"))?;
    assert!(
        s.alternate_screen(),
        "the dashboard is supposed to be holding the alternate screen"
    );
    Ok(t)
}

/// Every signal in the set: the terminal comes back, and the process still
/// reports as killed by that signal.
///
/// The second half matters as much as the first. A process that swallows
/// `SIGTERM` and exits 0 lies to `$?`, to a `timeout` wrapper and to a
/// supervisor — so `emulate_default_handler` re-raises with the default
/// disposition, and this is what pins that.
#[test]
fn a_signal_gives_the_terminal_back_and_still_kills_the_process() -> termlens::Result<()> {
    for (signal, name) in [
        (Signal::Term, "Terminated"),
        (Signal::Int, "Interrupt"),
        (Signal::Hup, "Hangup"),
    ] {
        let mut t = dashboard()?;
        t.signal(signal)?;
        let status = t.wait_exit()?;
        let after = t.screen();

        assert!(
            !after.alternate_screen(),
            "{name}: left the shell inside the alternate screen"
        );
        let (_, _, visible) = after.cursor();
        assert!(visible, "{name}: left the cursor hidden");
        // `contains`, not `==`: macOS spells this "Terminated: 15" and
        // Linux "Terminated". Asserting the exact string passed on Linux
        // and failed the macOS leg while the behaviour under test was
        // identical on both.
        let reported = status
            .signal()
            .unwrap_or_else(|| panic!("{name}: exited normally instead of dying, {status:?}"));
        assert!(
            reported.contains(name),
            "{name}: must still report as killed by the signal, got {reported:?}"
        );
    }
    Ok(())
}

/// The path that already worked, so that installing a signal handler
/// cannot quietly break it.
#[test]
fn quitting_normally_still_restores_and_exits_zero() -> termlens::Result<()> {
    for key in [Key::Char('q'), Key::Esc] {
        let mut t = dashboard()?;
        t.send(key)?;
        let status = t.wait_exit()?;
        assert!(status.success(), "{key:?}: exited with {status:?}");
        assert_eq!(status.signal(), None, "{key:?}: exited, not killed");
        assert!(
            !t.screen().alternate_screen(),
            "{key:?}: left the shell inside the alternate screen"
        );
        assert!(t.screen().cursor().2, "{key:?}: left the cursor hidden");
    }
    Ok(())
}
