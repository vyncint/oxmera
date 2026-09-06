//! Rendering the doctor report. Deterministic by contract: the same
//! `Report` always prints the same bytes — no clocks, no durations, no
//! absolute paths.

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

fn tool_line(name: &str, version: &Option<String>) -> String {
    match version {
        Some(v) => format!("  {name:<12} {v}"),
        None => format!("  {name:<12} not found"),
    }
}

/// Every crate's capability rows, in reading order: what a tensor is,
/// then what you can differentiate, build, train and save.
///
/// The order is this function's business; the *content* belongs to the
/// crate that implements it, so adding an op family is one line there and
/// no line here. See [`oxmera::tensor::CAPABILITIES`].
fn capabilities() -> Vec<(&'static str, &'static str)> {
    let mut rows: Vec<(&'static str, &'static str)> = Vec::new();
    rows.extend_from_slice(oxmera::tensor::CAPABILITIES);
    rows.extend_from_slice(oxmera::autograd::CAPABILITIES);
    rows.extend_from_slice(oxmera::nn::CAPABILITIES);
    rows.extend_from_slice(oxmera::optim::CAPABILITIES);
    rows
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
    if let Some(chip) = &r.host.chip {
        line(format!("  {:<12} {chip}", "chip"));
    }
    match (r.host.p_cores, r.host.e_cores) {
        (Some(p), Some(e)) => {
            line(format!(
                "  {:<12} {} ({p} performance + {e} efficiency)",
                "cpu cores",
                p + e
            ));
        }
        _ => line(format!("  {:<12} {}", "cpu cores", r.devices.cpu_threads)),
    }
    if let Some(gb) = r.host.memory_gb {
        line(format!("  {:<12} {gb:.1} GB unified", "memory"));
    }
    line(String::new());

    line("toolchain".into());
    line(tool_line("rustc", &r.toolchain.rustc));
    line(tool_line("cargo", &r.toolchain.cargo));
    line(String::new());

    line("devices".into());
    line(format!(
        "  cpu     available — {} threads (rayon)",
        r.devices.cpu_threads
    ));
    if r.devices.metal {
        let name = r.devices.metal_name.as_deref().unwrap_or("Metal device");
        match r.devices.metal_budget_gb {
            Some(gb) => line(format!(
                "  metal   {name} — unified memory budget {gb:.1} GB"
            )),
            None => line(format!("  metal   {name}")),
        }
    } else {
        line("  metal   not available on this host".into());
    }
    if r.devices.cuda {
        let name = r.devices.cuda_name.as_deref().unwrap_or("CUDA device");
        let cc = r.devices.cuda_cc.as_deref().unwrap_or("?");
        match r.devices.cuda_memory_gb {
            Some(gb) => line(format!(
                "  cuda    {name} — {gb:.1} GB, compute capability {cc}"
            )),
            None => line(format!("  cuda    {name} — compute capability {cc}")),
        }
    } else {
        line("  cuda    not available on this host".into());
    }
    line(String::new());

    line("capabilities".into());
    // Assembled from the crates that implement them, not written here.
    // The previous version was six string literals in this file, pinned by
    // three golden frames — so a release could add `eigh`, `einsum`, `f64`
    // and a whole CUDA backend without any of them appearing, and did.
    for (area, detail) in capabilities() {
        line(format!("  {area:<9} {detail}"));
    }
    line(String::new());

    line("try: `oxmera train --tui` — the live training dashboard".into());
    line(String::new());
    line("doctor: report complete".into());
    out
}
