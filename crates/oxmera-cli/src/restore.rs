//! Giving the terminal back, on every way out.
//!
//! A TUI borrows the terminal: the alternate screen, raw mode, the cursor.
//! The borrow has to be returned on **every** exit, and there are three —
//! the ordinary one, a panic, and a signal.
//!
//! Before 0.4.0 two of the three were covered. `Dashboard::finish` calls
//! `ratatui::restore()` on the way out, and `ratatui::init()` installs a
//! panic hook that does the same. A **signal** reached the same state by a
//! path with no guard at all: measured through a pty, `kill -TERM` on
//! `oxmera train --tui` left `Screen::alternate_screen()` true and the
//! cursor hidden, with nothing written to say so. The way out is to type
//! `reset` blind, which a first-time user does not know, and nothing on
//! screen says so — the failure is in what was *not* written.
//!
//! Whoever reaches for a signal is already having a bad time. This is a
//! *training* dashboard: the run is long by definition, so `Ctrl-C` in
//! another pane, `kill %1`, a `timeout` wrapper or a scheduler stopping
//! the job are all ordinary ways for this process to end.
//!
//! ## Why `signal-hook` rather than `libc::sigaction`
//!
//! `sigaction` is an unsafe call, and every crate in this workspace is
//! `forbid(unsafe_code)` or `deny(unsafe_code)`.
//!
//! `signal-hook` registers safely and — the part that matters — is
//! *already in the tree*: crossterm pulls it through ratatui at the same
//! version pinned here. So this adds an import, not a dependency: no new
//! third-party code, no new licence to review, no new supply chain, and
//! nothing new for `deny.toml` to have an opinion about.
//!
//! The work happens on a thread rather than in a handler, which is what
//! makes it safe to do anything at all: a real signal handler may call
//! only async-signal-safe functions, and writing escape sequences through
//! Rust's stdout is not one of them.

use std::io;

use crossterm::cursor::Show;
use crossterm::execute;

/// Give the terminal back: alternate screen off, raw mode off, cursor
/// shown.
///
/// Best-effort by construction. This runs on the way out of a process that
/// may already be failing, and on a terminal that may already have gone;
/// an error here must not mask the reason we are leaving.
pub fn terminal() {
    ratatui::restore();
    // Explicitly, and last. Leaving the alternate screen restores *that*
    // screen's cursor state, which is not necessarily the shell's — and a
    // hidden cursor is the one piece of this a user cannot see is missing
    // and cannot guess the cure for. The reproduction in #40 measured
    // `cursor() == (44, 36, false)` after SIGTERM: the third field is
    // `visible`.
    let _ = execute!(io::stdout(), Show);
}

/// Restore the terminal on `SIGINT`, `SIGTERM` and `SIGHUP`, then die of
/// the signal.
///
/// Re-raising with the default disposition matters: a process killed by a
/// signal must still *report* as killed by that signal, or a shell's `$?`,
/// a `timeout` wrapper and a supervisor all learn the wrong thing about
/// why it stopped. `emulate_default_handler` does exactly that, and the
/// tests assert the exit status still carries the signal name.
///
/// `SIGHUP` belongs in the set: a closed terminal emulator is exactly when
/// nobody is left to type `reset`.
///
/// Ctrl-C *typed into* the dashboard is a key, handled by the event loop,
/// and does not come through here — the damage always needed an actual
/// signal.
///
/// Failing to register is not worth failing the run over: the terminal is
/// no worse off than it was before 0.4.0, and the user asked to train a
/// model, not to install a signal handler.
#[cfg(unix)]
pub fn on_signal() {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
    let Ok(mut signals) = signal_hook::iterator::Signals::new([SIGINT, SIGTERM, SIGHUP]) else {
        return;
    };
    std::thread::spawn(move || {
        for signal in signals.forever() {
            terminal();
            // Unregisters our handler and re-raises, so the exit status is
            // death-by-signal rather than a plain code.
            let _ = signal_hook::low_level::emulate_default_handler(signal);
        }
    });
}

/// No-op off Unix.
///
/// Windows has no POSIX signals: a console application is torn down
/// through a control handler with a different lifetime and a different
/// contract, and pretending otherwise here would mean claiming a guarantee
/// that is not installed. The panic hook `ratatui::init()` installs covers
/// the other unguarded exit on every platform.
#[cfg(not(unix))]
pub fn on_signal() {}
