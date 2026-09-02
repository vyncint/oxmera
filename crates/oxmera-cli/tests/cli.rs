//! The front door: help is a successful request on stdout; a typo is a
//! failure on stderr. The two must stay distinguishable to scripts
//! (issue #21).
use std::process::Command;

fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_oxmera"))
        .args(args)
        .output()
        .expect("spawn oxmera");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn help_is_stdout_and_exit_zero_in_every_spelling() {
    for flag in ["--help", "-h", "help"] {
        let (code, stdout, stderr) = run(&[flag]);
        assert_eq!(code, 0, "{flag}: exit code");
        assert!(
            stdout.starts_with("usage: oxmera"),
            "{flag}: stdout was {stdout:?}"
        );
        assert!(
            stdout.contains("--help"),
            "{flag}: usage must mention the help flag"
        );
        assert!(stderr.is_empty(), "{flag}: stderr was {stderr:?}");
    }
}

#[test]
fn unknown_subcommand_is_stderr_and_exit_one() {
    for args in [&["frobnicate"][..], &[]] {
        let (code, stdout, stderr) = run(args);
        assert_eq!(code, 1, "{args:?}: exit code");
        assert!(stdout.is_empty(), "{args:?}: stdout was {stdout:?}");
        assert!(
            stderr.starts_with("usage: oxmera"),
            "{args:?}: stderr was {stderr:?}"
        );
    }
}

#[test]
fn version_is_stdout_and_exit_zero() {
    let (code, stdout, stderr) = run(&["--version"]);
    assert_eq!(code, 0);
    assert_eq!(
        stdout.trim(),
        format!("oxmera {}", env!("CARGO_PKG_VERSION"))
    );
    assert!(stderr.is_empty());
}
