//! `termlens-cli` — the command that ships beside the harness this crate
//! already tests with.
//!
//! The rest of the suite asks whether `oxmera` draws the right thing. This
//! asks what a maintainer does *after* it draws the wrong thing, at a shell
//! prompt, with no test to edit: point `termlens inspect` at the binary,
//! and read the committed goldens with `termlens diff` and `termlens
//! render`.
//!
//! Two things make that worth a test here rather than upstream.
//!
//! 1. **The goldens are the CLI's input format.** `tests/golden/*.txt` are
//!    written by `Screen::to_string()` through this crate's `normalize`,
//!    committed, and compared as strings. Nothing checked that they are
//!    still *saved screens* — so `termlens render --svg
//!    tests/golden/tui-replay-100x45.txt`, the way a frame gets into a bug
//!    report, could have quietly stopped working.
//! 2. **The two paths must agree.** `inspect` drives the real `oxmera`
//!    through its own PTY, with its own defaults. If its picture and the
//!    golden `cargo test` compares against ever differ, one of the two is
//!    lying about what the program draws.
//!
//! `#[ignore]`d by default: these need `termlens-cli` on the machine, and a
//! `cargo test` that quietly `cargo install`s something is a surprise no
//! published crate should spring on a contributor — `oxmera-cli` is on
//! crates.io, and AGENTS.md requires a fresh clone to build and test with
//! nothing exotic. `stress.yml` runs them by name.
//!
//! ```sh
//! cargo test -p oxmera-cli --test termlens_cli -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// The termlens version this crate is measured against, read from the
/// committed workspace lockfile so the tool and the library can never be
/// two different releases.
fn version_under_test() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let lock =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
                .expect("Cargo.lock is committed at the workspace root");
        let mut lines = lock.lines();
        while let Some(line) = lines.next() {
            if line.trim() == "name = \"termlens\"" {
                for next in lines.by_ref() {
                    if let Some(rest) = next.trim().strip_prefix("version = \"") {
                        return rest.trim_end_matches('"').to_owned();
                    }
                }
            }
        }
        panic!("no termlens version in Cargo.lock");
    })
}

/// The `termlens` binary: `$TERMLENS_CLI` if the environment provides one,
/// otherwise installed once into the workspace `target/` at the version
/// under test. `target/` is gitignored, so this leaves no trace in a clone.
fn cli() -> &'static PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        if let Some(given) = std::env::var_os("TERMLENS_CLI") {
            return PathBuf::from(given);
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target")
            .join("termlens-cli");
        let bin = root
            .join("bin")
            .join(format!("termlens{}", std::env::consts::EXE_SUFFIX));
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let status = Command::new(cargo)
            .args(["install", "termlens-cli", "--version", version_under_test()])
            .args(["--locked", "--root"])
            .arg(&root)
            .status()
            .expect("cargo install termlens-cli");
        assert!(
            status.success(),
            "cargo install termlens-cli --version {} failed. It is published \
             alongside the library; if this version of termlens exists on \
             crates.io and termlens-cli does not, the two releases went out \
             of lockstep.",
            version_under_test()
        );
        bin
    })
}

fn run(args: &[&str]) -> Output {
    Command::new(cli())
        .args(args)
        .output()
        .expect("run termlens")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn golden(name: &str) -> PathBuf {
    golden_dir().join(name)
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn path(p: &Path) -> String {
    p.to_str().expect("utf-8 path").to_owned()
}

/// A temporary file, named after this process so parallel runs cannot
/// collide.
fn scratch(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("oxmera-{}-{name}", std::process::id()));
    std::fs::write(&path, contents).expect("write the scratch screen");
    path
}

#[test]
#[ignore = "needs termlens-cli; run with --ignored (stress.yml does)"]
fn the_tool_and_the_library_are_one_release() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        format!("termlens {}", version_under_test()),
        "the installed CLI is not the version this crate tests against"
    );
}

/// Every committed golden is still a saved screen a person can render.
///
/// `assert_golden` compares them as strings, so a change to `normalize`
/// could leave them readable by nothing but this crate — and the goldens
/// are exactly what you want to turn into an SVG when a frame changed and
/// you are explaining it to somebody.
#[test]
#[ignore = "needs termlens-cli; run with --ignored (stress.yml does)"]
fn every_committed_golden_renders_from_a_shell() {
    let mut seen = 0;
    for entry in std::fs::read_dir(golden_dir()).expect("tests/golden exists") {
        let file = entry.expect("readable entry").path();
        if file.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let name = file.file_name().unwrap().to_string_lossy().into_owned();

        // The text rendering is the format itself, so it must read back as
        // the same picture: parse, render, parse again, no difference.
        let text = run(&["render", "--text", &path(&file)]);
        assert!(
            text.status.success(),
            "{name}: render --text failed: {}",
            String::from_utf8_lossy(&text.stderr)
        );
        let round = scratch(
            &format!("{name}.round"),
            &String::from_utf8_lossy(&text.stdout),
        );
        let back = run(&["diff", "--color", "never", &path(&file), &path(&round)]);
        assert_eq!(
            back.status.code(),
            Some(0),
            "{name}: the golden does not survive a render/parse round trip:\n{}",
            String::from_utf8_lossy(&back.stdout)
        );

        // And the image, which is what goes into an issue.
        let svg = run(&["render", "--svg", &path(&file)]);
        assert!(svg.status.success(), "{name}: render --svg failed");
        let body = String::from_utf8_lossy(&svg.stdout);
        assert!(body.starts_with("<svg "), "{name}: an SVG document");
        assert!(
            body.contains("oxmera"),
            "{name}: with the frame's own text in it"
        );

        let _ = std::fs::remove_file(&round);
        seen += 1;
    }
    assert_eq!(seen, 5, "the five goldens this crate commits");
}

/// `diff` between two committed goldens, and all three of its exit codes.
///
/// The three environment shapes exist precisely so a change to the report
/// can be read one shape at a time, and this is how you read one without a
/// test: exit 0 is "the same picture", exit 1 is "here is what moved", exit
/// 2 is "that was not a screen".
#[test]
#[ignore = "needs termlens-cli; run with --ignored (stress.yml does)"]
fn diff_tells_the_environment_shapes_apart() {
    let metal = path(&golden("doctor-metal-100x45.txt"));
    let cuda = path(&golden("doctor-cuda-100x45.txt"));

    let same = run(&["diff", "--color", "never", &metal, &metal]);
    assert_eq!(same.status.code(), Some(0), "a screen equals itself");
    assert!(String::from_utf8_lossy(&same.stdout).contains("no difference"));

    let out = run(&["diff", "--color", "never", &metal, &cuda]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "an Apple machine and an NVIDIA one are different reports"
    );
    let rendered = String::from_utf8_lossy(&out.stdout);
    assert!(rendered.contains("size: 100x45"), "the header:\n{rendered}");
    assert!(
        rendered.contains("rows unchanged"),
        "and a count of what did not move:\n{rendered}"
    );
    // The device rows are the point of the two fixtures, so they must be
    // among the rows it names as changed.
    assert!(
        rendered.contains("Apple M3 Pro (fixture)") && rendered.contains("NVIDIA A10G (fixture)"),
        "the device rows are what differs:\n{rendered}"
    );

    // Two geometries of the same frame: a size delta, reported rather than
    // refused.
    let wide = path(&golden("tui-replay-100x45.txt"));
    let narrow = path(&golden("tui-replay-80x30.txt"));
    let resized = run(&["diff", "--color", "never", &wide, &narrow]);
    assert_eq!(resized.status.code(), Some(1));
    let text = String::from_utf8_lossy(&resized.stdout);
    assert!(
        text.contains("100x45 → 80x30"),
        "the size delta is named:\n{text}"
    );

    // Exit 2 is the code a script must not confuse with "they differ".
    let manifest = path(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"));
    let broken = run(&["diff", "--color", "never", &manifest, &manifest]);
    assert_eq!(
        broken.status.code(),
        Some(2),
        "a file that is not a saved screen is an error, not a difference"
    );
    assert!(
        String::from_utf8_lossy(&broken.stderr).contains("could not parse a saved screen"),
        "and says so on stderr"
    );
}

/// `inspect` drives the real `oxmera`, and gets the committed picture.
///
/// This is the assertion that ties the two paths together. `inspect` builds
/// its own PTY with its own defaults and none of this crate's helpers; if
/// its screen and the golden `cargo test` compares against ever disagree,
/// one of them is wrong about what `oxmera` draws.
#[test]
#[ignore = "needs termlens-cli; run with --ignored (stress.yml does)"]
fn inspect_drives_oxmera_and_gets_the_committed_picture() {
    // `doctor` prints and exits, so inspect reports its exit status.
    let out = Command::new(cli())
        .args(["inspect", "--size", "100x45"])
        .arg(env!("CARGO_BIN_EXE_oxmera"))
        .args(["doctor", "--fixture"])
        .arg(fixture("metal.toml"))
        .output()
        .expect("run inspect");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8_lossy(&out.stdout);
    // The trailer reports the program's own exit, on **stderr** since
    // termlens 0.11 (termlens#340) — so stdout is the screen alone and
    // needs no separating here.
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--- exited: exit code 0 ---"),
        "the trailer reports the program's own exit: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !printed.contains("--- "),
        "stdout is a saved screen:\n{printed}"
    );

    // The header plus exactly `rows` lines, which is what `diff` reads.
    let grid: String = printed
        .lines()
        .take(46)
        .map(|line| format!("{line}\n"))
        .collect();
    let seen = scratch("inspect-doctor.snap", &grid);
    let against = run(&[
        "diff",
        "--color",
        "never",
        &path(&seen),
        &path(&golden("doctor-metal-100x45.txt")),
    ]);
    assert_eq!(
        against.status.code(),
        Some(0),
        "the picture `termlens inspect` gets from oxmera is not the one the \
         golden pins:\n{}",
        String::from_utf8_lossy(&against.stdout)
    );
    let _ = std::fs::remove_file(&seen);

    // The dashboard never exits on its own — it waits for `q` — so inspect
    // snapshots it after the output goes quiet and reports it still running.
    // That distinction is the whole reason `--idle` exists. Which "still
    // running" it is depends on what ended the wait: since termlens 0.11.3
    // (termlens#374) a TUI that goes quiet for `--idle` reads `still running
    // (killed on exit)`, and only one that keeps printing until `--timeout`
    // says `at the deadline`. Both mean it did not exit, which is all this
    // asserts.
    let out = Command::new(cli())
        .args([
            "inspect",
            "--size",
            "100x45",
            "--idle",
            "600",
            "--timeout",
            "30",
        ])
        .arg(env!("CARGO_BIN_EXE_oxmera"))
        .args(["train", "--replay"])
        .arg(fixture("train-replay.toml"))
        .output()
        .expect("run inspect");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8_lossy(&out.stdout);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("still running"),
        "a TUI holds the terminal, so inspect reports it still running rather \
         than an exit, on stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !printed.contains("--- "),
        "stdout is a saved screen:\n{printed}"
    );
    let grid: String = printed
        .lines()
        .take(46)
        .map(|line| format!("{line}\n"))
        .collect();
    let seen = scratch("inspect-tui.snap", &grid);
    let against = run(&[
        "diff",
        "--color",
        "never",
        &path(&seen),
        &path(&golden("tui-replay-100x45.txt")),
    ]);
    assert_eq!(
        against.status.code(),
        Some(0),
        "the dashboard `termlens inspect` renders is not the golden frame:\n{}",
        String::from_utf8_lossy(&against.stdout)
    );
    let _ = std::fs::remove_file(&seen);
}
