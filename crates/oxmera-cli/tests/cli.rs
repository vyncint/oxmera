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

/// Every capability row has to fit a narrow terminal.
///
/// The doctor goldens are 100 columns; a row longer than that wraps and
/// costs two rows of the frame, and the rows are written in four other
/// crates where nobody is looking at a terminal. 80 columns is the real
/// floor a terminal tool should respect, so that is what this pins.
#[test]
fn capability_rows_fit_a_narrow_terminal() {
    let rows: Vec<(&str, &str)> = oxmera::tensor::CAPABILITIES
        .iter()
        .chain(oxmera::autograd::CAPABILITIES)
        .chain(oxmera::nn::CAPABILITIES)
        .chain(oxmera::optim::CAPABILITIES)
        .copied()
        .collect();
    assert!(!rows.is_empty(), "no crate declared any capability");
    for (area, detail) in rows {
        // "  " + area padded to 9 + " " = 12 characters of prefix.
        let rendered = 12 + detail.chars().count();
        assert!(
            rendered <= 80,
            "the `{area}` row renders {rendered} columns wide and would wrap \
             an 80-column terminal: {detail:?}"
        );
        assert!(
            area.chars().count() <= 9,
            "the `{area}` label is wider than the column it is padded to"
        );
    }
}

/// `oxmera doctor` must name every op family the release ships.
///
/// The list below is deliberately hand-written and deliberately in the
/// *test*: it is the second opinion. The capability rows now live beside
/// the code that implements them, which stops them going stale by
/// accident — this stops a whole family being added to a crate that has
/// no row at all, which is how CUDA, `f64`, the linear algebra and
/// `einsum` were all absent from the report at 0.3.0.
///
/// Adding a family here when you add it to a crate is the point. If this
/// fails, do not delete the entry — add the row.
#[test]
fn doctor_names_every_family_this_release_ships() {
    // Through the binary, because the report a user reads is the one the
    // binary prints — not a function a test can call with different
    // arguments. `--fixture` keeps it hermetic.
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/no-gpu.toml");
    let (code, report, stderr) = run(&["doctor", "--fixture", &fixture.to_string_lossy()]);
    assert_eq!(code, 0, "doctor failed: {stderr}");

    // Whole words, not substrings. The first version of this test used
    // `report.contains(family)` and passed with `eigh` deleted from the
    // linalg row — because `weights` contains `eigh`. A guard that cannot
    // fail is worse than no guard: it is a green light nobody will
    // re-examine.
    let words: std::collections::HashSet<&str> = report
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .filter(|w| !w.is_empty())
        .collect();

    for family in [
        "matmul",
        "f64",
        "cholesky",
        "eigh",
        "logdet",
        "einsum",
        "index_select",
        "index_add",
        "autograd",
        "Conv2d",
        "LayerNorm",
        "safetensors",
        "AdamW",
        "per-group",
    ] {
        assert!(
            words.contains(family),
            "`oxmera doctor` never mentions {family:?}, which this release ships.\n\
             Add it to the CAPABILITIES const of the crate that implements it.\n\
             --- report ---\n{report}"
        );
    }
}
