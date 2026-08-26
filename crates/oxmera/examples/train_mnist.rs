//! Train a classifier with oxmera: MNIST when the IDX files are present,
//! a synthetic two-spirals fallback otherwise (so the example always runs
//! offline).
//!
//! ```text
//! cargo run --release -p oxmera --example train_mnist -- --device cpu
//! cargo run --release -p oxmera --example train_mnist -- --device metal --epochs 3
//! ```
//!
//! MNIST: place `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`
//! (unzipped) under `./data/mnist/`. For the live dashboard, use the CLI:
//! `oxmera train --tui`.

use oxmera::nn::{CrossEntropyLoss, Linear, Module, Param, Sequential};
use oxmera::optim::{Adam, Optimizer};
use oxmera::{Device, Result, Shape, Tensor};

struct Activation;

impl Module for Activation {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }
    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Param)> {
        Vec::new()
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("train_mnist: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut device = Device::Cpu;
    let mut epochs = 3usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--device" => {
                device = match it.next().map(String::as_str) {
                    Some("metal") => Device::Metal { index: 0 },
                    _ => Device::Cpu,
                }
            }
            "--epochs" => {
                epochs = it.next().and_then(|v| v.parse().ok()).unwrap_or(3);
            }
            _ => {}
        }
    }
    let devices = oxmera::init();
    println!("devices: {devices:?}; training on {device:?}");

    let (x, y, in_dim, classes, name) = load_dataset()?;
    let n = x.dims()[0];
    println!("dataset: {name} — {n} samples, {in_dim} features, {classes} classes");

    let model = Sequential::new()
        .push(Linear::new(in_dim, 128, 1))
        .push(Activation)
        .push(Linear::new(128, 64, 2))
        .push(Activation)
        .push(Linear::new(64, classes, 3));
    let mut opt = Adam::new(model.parameters(), 1e-3);
    let loss_fn = CrossEntropyLoss;

    let batch = 64usize;
    let batches = n / batch;
    let x_dev = x.to_device(device)?;
    for epoch in 0..epochs {
        let started = std::time::Instant::now();
        let mut epoch_loss = 0.0;
        for b in 0..batches {
            let xb = x_dev.narrow(0, b * batch, batch)?;
            let yb = y.narrow(0, b * batch, batch)?;
            opt.zero_grad();
            let logits = model.forward(&xb)?.to_device(Device::Cpu)?;
            let loss = loss_fn.forward(&logits, &yb)?;
            epoch_loss += loss.get_f32(&[])?;
            loss.backward()?;
            opt.step()?;
        }
        let logits = oxmera::no_grad(|| model.forward(&x_dev))?.to_device(Device::Cpu)?;
        let pred = logits.argmax(1, false)?.to_vec_i64()?;
        let want = y.to_vec_i64()?;
        let acc = pred.iter().zip(&want).filter(|(a, b)| a == b).count() as f32 / n as f32;
        println!(
            "epoch {:>2}/{}: loss {:.4}  accuracy {:.2}%  ({:.1}s, {:.0} samples/s)",
            epoch + 1,
            epochs,
            epoch_loss / batches as f32,
            acc * 100.0,
            started.elapsed().as_secs_f32(),
            n as f32 / started.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

/// MNIST if present, spirals otherwise.
fn load_dataset() -> Result<(Tensor, Tensor, usize, usize, &'static str)> {
    match load_mnist() {
        Some((x, y)) => Ok((x, y, 784, 10, "MNIST")),
        None => {
            let (x, y) = spirals(512);
            Ok((
                x,
                y,
                2,
                2,
                "two spirals (MNIST files not found under ./data/mnist)",
            ))
        }
    }
}

fn load_mnist() -> Option<(Tensor, Tensor)> {
    let images = std::fs::read("data/mnist/train-images-idx3-ubyte").ok()?;
    let labels = std::fs::read("data/mnist/train-labels-idx1-ubyte").ok()?;
    if images.len() < 16 || labels.len() < 8 {
        return None;
    }
    let n = u32::from_be_bytes([images[4], images[5], images[6], images[7]]) as usize;
    let n = n.min(10_000); // keep the example brisk
    let xs: Vec<f32> = images[16..16 + n * 784]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();
    let ys: Vec<i64> = labels[8..8 + n].iter().map(|&b| b as i64).collect();
    Some((
        Tensor::from_vec_f32(xs, Shape::from([n, 784])).ok()?,
        Tensor::from_vec_i64(ys, Shape::from([n])).ok()?,
    ))
}

fn spirals(n_per_class: usize) -> (Tensor, Tensor) {
    let noise = Tensor::randn_with_seed([n_per_class * 4], 7)
        .to_vec_f32()
        .expect("cpu");
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for class in 0..2i64 {
        for i in 0..n_per_class {
            let t = i as f32 / n_per_class as f32 * 3.5 + 0.2;
            let angle = t * 3.0 + class as f32 * std::f32::consts::PI;
            let idx = (class as usize * n_per_class + i) * 2 % (n_per_class * 4);
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
