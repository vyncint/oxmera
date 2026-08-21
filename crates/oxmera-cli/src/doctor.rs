//! Rendering the doctor report. Deterministic by contract: the same
//! `Report` always prints the same bytes.

use crate::probe;
use crate::report::Report;

pub fn run(args: &[String]) -> Result<(), String> {
    let report = match args {
        [] => probe::probe(),
        [flag, path] if flag == "--fixture" => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read fixture {path}: {e}"))?;
            toml::from_str(&text).map_err(|e| format!("cannot parse fixture {path}: {e}"))?
        }
        _ => return Err("usage: oxmera doctor [--fixture <path>]".into()),
    };
    print!("{}", render(&report));
    Ok(())
}

fn yes_no(present: bool) -> &'static str {
    if present { "yes" } else { "no" }
}

fn tool_line(name: &str, version: &Option<String>) -> String {
    match version {
        Some(v) => format!("  {name:<12} {v}"),
        None => format!("  {name:<12} not found"),
    }
}

fn render(r: &Report) -> String {
    let mut out = String::new();
    let mut line = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };

    line("oxmera doctor".into());
    line("=============".into());
    line(String::new());
    line(format!("host: {} / {}", r.os, r.arch));
    line(String::new());

    line("toolchain".into());
    line(tool_line("rustc", &r.toolchain.rustc));
    line(tool_line("cargo", &r.toolchain.cargo));
    line(tool_line("just", &r.toolchain.just));
    line(tool_line("reconverge", &r.toolchain.reconverge));
    line(tool_line("launchbound", &r.toolchain.launchbound));
    line(format!(
        "  {:<12} {}",
        "container",
        yes_no(r.toolchain.container)
    ));
    line(String::new());

    let tier0 = r.toolchain.rustc.is_some() && r.toolchain.cargo.is_some();
    let tier0_gate = tier0 && r.toolchain.reconverge.is_some();
    line("tiers".into());
    line(format!(
        "  tier 0  edit / check / CPU backend        {}",
        if tier0 {
            "available"
        } else {
            "MISSING rustc/cargo"
        }
    ));
    line(format!(
        "  tier 0  convergence gate (reconverge)     {}",
        if tier0_gate {
            "available"
        } else {
            "install cargo-reconverge + reconverge-driver"
        }
    ));
    line(format!(
        "  tier 1  container PTX (cargo oxide)       {}",
        if r.toolchain.container {
            "available"
        } else {
            "no container runtime found"
        }
    ));
    line("  tier 2  NVIDIA execution                  metered — approved GPU sessions only".into());
    line(String::new());

    line("backend paths".into());
    line("  cpu     always available — the correctness ground truth".into());
    line(format!(
        "  metal   {}  (no convergence gate exists on this path)",
        if r.gpu.metal {
            "plausible on this host"
        } else {
            "not on this host"
        }
    ));
    line(format!(
        "  cuda    {}  (verdicts and timings are per-part)",
        if r.gpu.cuda {
            "toolkit visible"
        } else {
            "no toolkit here — tier 1/2 territory"
        }
    ));
    line(String::new());

    line("exercise ladder".into());
    if !r.ladder_found {
        line("  no exercises/manifest.toml found from here".into());
    } else {
        let solved = r.ladder.iter().filter(|x| x.status == "solved").count();
        let todo = r.ladder.iter().filter(|x| x.status == "todo").count();
        let planned = r.ladder.iter().filter(|x| x.status == "planned").count();
        line(format!("  {solved} solved, {todo} todo, {planned} planned"));
        for rung in &r.ladder {
            line(format!(
                "  [{}] {:<4} {}",
                status_mark(&rung.status),
                rung.id,
                rung.name
            ));
        }
    }
    line(String::new());
    line("doctor: report complete".into());
    out
}

fn status_mark(status: &str) -> &'static str {
    match status {
        "solved" => "x",
        "todo" => " ",
        _ => ".",
    }
}
