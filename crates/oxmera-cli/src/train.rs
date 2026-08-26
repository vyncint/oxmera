//! `oxmera train`: a real training run on a deterministic synthetic
//! dataset (two interleaved spirals), on CPU or Metal, with a plain-text
//! reporter or the live TUI dashboard — plus a `--replay` mode that renders
//! the dashboard from a recorded fixture, which is what the termlens
//! goldens drive (no clocks, no randomness, no real training in tests).

use std::time::Instant;

use oxmera::nn::{CrossEntropyLoss, Linear, Module, Param, Sequential};
use oxmera::optim::{Adam, Optimizer};
use oxmera::{Device, Shape, Tensor};
use serde::Deserialize;

use crate::tui::{self, DashState};

/// One epoch's recorded metrics.
#[derive(Debug, Clone, Deserialize)]
pub struct EpochMetrics {
    /// Mean training loss.
    pub loss: f32,
    /// Training-set accuracy in `[0, 1]`.
    pub accuracy: f32,
    /// Samples per second over the epoch.
    pub throughput: f32,
}

/// A recorded run for `--replay`.
#[derive(Debug, Deserialize)]
pub struct Replay {
    /// Device label shown in the header.
    pub device: String,
    /// Model label shown in the header.
    pub model: String,
    /// Batches per epoch (drives the batch gauge).
    pub batches_per_epoch: usize,
    /// Unified-memory label (empty hides the field).
    #[serde(default)]
    pub memory: String,
    /// Per-epoch metrics, in order.
    #[serde(rename = "epoch")]
    pub epochs: Vec<EpochMetrics>,
}

struct Args {
    device: Device,
    epochs: usize,
    tui: bool,
    replay: Option<String>,
    seed: u64,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut out = Args {
        device: Device::Cpu,
        epochs: 12,
        tui: false,
        replay: None,
        seed: 7,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tui" => out.tui = true,
            "--device" => match it.next().map(String::as_str) {
                Some("cpu") => out.device = Device::Cpu,
                Some("metal") => out.device = Device::Metal { index: 0 },
                other => return Err(format!("--device expects cpu|metal, got {other:?}")),
            },
            "--epochs" => {
                out.epochs = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--epochs expects a number")?;
            }
            "--seed" => {
                out.seed = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or("--seed expects a number")?;
            }
            "--replay" => {
                out.replay = Some(it.next().ok_or("--replay expects a path")?.clone());
                out.tui = true;
            }
            other => return Err(format!("unknown train argument {other}")),
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Result<(), String> {
    let args = parse(args)?;

    if let Some(path) = &args.replay {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let replay: Replay =
            toml::from_str(&text).map_err(|e| format!("cannot parse {path}: {e}"))?;
        return tui::run_replay(&replay).map_err(|e| format!("tui: {e}"));
    }

    oxmera::init();
    if oxmera::runtime::backend_for(args.device).is_err() {
        return Err(format!(
            "device {:?} is not available on this machine",
            args.device
        ));
    }
    train_real(&args).map_err(|e| format!("train: {e}"))
}

/// Two interleaved spirals: 2 features, 2 classes, deterministic.
fn spirals(n_per_class: usize, seed: u64) -> (Tensor, Tensor) {
    let noise = Tensor::randn_with_seed([2 * n_per_class * 2], seed)
        .to_vec_f32()
        .expect("cpu");
    let mut xs = Vec::with_capacity(n_per_class * 4);
    let mut ys = Vec::with_capacity(n_per_class * 2);
    for class in 0..2i64 {
        for i in 0..n_per_class {
            let t = i as f32 / n_per_class as f32 * 3.5 + 0.2;
            let angle = t * 3.0 + class as f32 * std::f32::consts::PI;
            let idx = (class as usize * n_per_class + i) * 2;
            xs.push(t * angle.cos() + 0.08 * noise[idx]);
            xs.push(t * angle.sin() + 0.08 * noise[idx + 1]);
            ys.push(class);
        }
    }
    let n = ys.len();
    (
        Tensor::from_vec_f32(xs, Shape::from([n, 2])).expect("shape"),
        Tensor::from_vec_i64(ys, Shape::from([n])).expect("shape"),
    )
}

fn accuracy(logits: &Tensor, targets: &Tensor) -> oxmera::Result<f32> {
    let pred = logits.argmax(1, false)?.to_vec_i64()?;
    let want = targets.to_vec_i64()?;
    let hits = pred.iter().zip(&want).filter(|(a, b)| a == b).count();
    Ok(hits as f32 / want.len() as f32)
}

fn train_real(args: &Args) -> oxmera::Result<()> {
    let (x_all, y_all) = spirals(128, args.seed);
    let n = x_all.dims()[0];
    let batch = 32usize;
    let batches = n / batch;

    let model = Sequential::new()
        .push(Linear::new(2, 32, args.seed + 1))
        .push(Activation)
        .push(Linear::new(32, 32, args.seed + 2))
        .push(Activation)
        .push(Linear::new(32, 2, args.seed + 3));
    let mut opt = Adam::new(model.parameters(), 0.01);
    let loss_fn = CrossEntropyLoss;

    let device_label = args.device.kind_name().to_string();
    let mut dash = args
        .tui
        .then(|| {
            tui::start(DashState::new(
                &device_label,
                "mlp 2-32-32-2",
                args.epochs,
                batches,
                memory_label(args.device),
            ))
        })
        .transpose()
        .map_err(|e| oxmera::Error::Io {
            op: "tui",
            detail: e,
        })?;

    let x_dev = x_all.to_device(args.device)?;
    for epoch in 0..args.epochs {
        let started = Instant::now();
        let mut epoch_loss = 0.0f32;
        for b in 0..batches {
            let xb = x_dev.narrow(0, b * batch, batch)?;
            let yb = y_all.narrow(0, b * batch, batch)?;
            opt.zero_grad();
            let logits = model.forward(&xb)?;
            let loss = loss_fn.forward(&logits.to_device(Device::Cpu)?, &yb)?;
            epoch_loss += loss.get_f32(&[])?;
            loss.backward()?;
            opt.step()?;
            if let Some(d) = &mut dash {
                d.batch_tick(epoch, b + 1).map_err(|e| oxmera::Error::Io {
                    op: "tui",
                    detail: e,
                })?;
            }
        }
        let logits = oxmera::no_grad(|| model.forward(&x_dev))?;
        let acc = accuracy(&logits.to_device(Device::Cpu)?, &y_all)?;
        let throughput = (n as f32) / started.elapsed().as_secs_f32().max(1e-6);
        let metrics = EpochMetrics {
            loss: epoch_loss / batches as f32,
            accuracy: acc,
            throughput,
        };
        match &mut dash {
            Some(d) => d.epoch_done(metrics).map_err(|e| oxmera::Error::Io {
                op: "tui",
                detail: e,
            })?,
            None => println!(
                "epoch {:>3}/{}: loss {:.4}  accuracy {:.1}%  {:.0} samples/s",
                epoch + 1,
                args.epochs,
                epoch_loss / batches as f32,
                acc * 100.0,
                throughput
            ),
        }
    }
    if let Some(d) = dash {
        d.finish().map_err(|e| oxmera::Error::Io {
            op: "tui",
            detail: e,
        })?;
    } else {
        println!("training complete on {device_label}");
    }
    Ok(())
}

fn memory_label(device: Device) -> String {
    match device {
        #[cfg(target_os = "macos")]
        Device::Metal { .. } => match oxmera::metal::device_summary() {
            Some((_, budget, used)) => format!(
                "{:.1} / {:.1} GB unified",
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                budget as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
            None => String::new(),
        },
        _ => String::new(),
    }
}

/// Parameter-free tanh activation for the demo pipeline.
struct Activation;

impl Module for Activation {
    fn forward(&self, input: &Tensor) -> oxmera::Result<Tensor> {
        input.tanh()
    }
    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Param)> {
        Vec::new()
    }
}
