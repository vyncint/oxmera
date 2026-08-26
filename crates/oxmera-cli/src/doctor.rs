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
    line("  cuda    deferred — no backend in this build".into());
    line(String::new());

    line("capabilities".into());
    line(
        "  tensor    f32 strided views, broadcasting, batched matmul, cross-device transfer".into(),
    );
    line("  autograd  reverse-mode tape, finite-difference verified".into());
    line("  nn        Linear Conv2d Embedding LayerNorm BatchNorm2d Dropout Sequential".into());
    line("  losses    MSE CrossEntropy BCEWithLogits".into());
    line("  optim     SGD Adam AdamW RMSprop".into());
    line("  weights   safetensors save/load".into());
    line(String::new());

    line("try: `oxmera train --tui` — the live training dashboard".into());
    line(String::new());
    line("doctor: report complete".into());
    out
}
