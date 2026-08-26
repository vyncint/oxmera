//! The training dashboard: loss and accuracy sparklines, epoch/batch
//! gauges, throughput and memory readouts.
//!
//! Two drivers share one renderer:
//! - live training (`train --tui`) pushes real metrics as they happen;
//! - replay (`train --replay fixture.toml`) renders a recorded run frame
//!   by frame with no clocks and no randomness — the mode the termlens
//!   goldens capture.

use crossterm::event::{self, Event, KeyCode};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};
use ratatui::{DefaultTerminal, Frame};

use crate::train::{EpochMetrics, Replay};

/// Everything one dashboard frame needs.
pub struct DashState {
    device: String,
    model: String,
    epochs_total: usize,
    batches_total: usize,
    memory: String,
    epoch: usize,
    batch: usize,
    history: Vec<EpochMetrics>,
    done: bool,
}

impl DashState {
    /// A fresh dashboard for a run of `epochs_total` x `batches_total`.
    pub fn new(
        device: &str,
        model: &str,
        epochs_total: usize,
        batches_total: usize,
        memory: String,
    ) -> Self {
        Self {
            device: device.to_string(),
            model: model.to_string(),
            epochs_total,
            batches_total,
            memory,
            epoch: 0,
            batch: 0,
            history: Vec::new(),
            done: false,
        }
    }

    fn latest(&self) -> Option<&EpochMetrics> {
        self.history.last()
    }
}

/// A live dashboard bound to a real terminal.
pub struct Dashboard {
    terminal: DefaultTerminal,
    state: DashState,
}

/// Enter the alternate screen and draw the initial (empty) dashboard.
pub fn start(state: DashState) -> Result<Dashboard, String> {
    let mut terminal = ratatui::init();
    terminal
        .draw(|f| draw(f, &state))
        .map_err(|e| e.to_string())?;
    Ok(Dashboard { terminal, state })
}

impl Dashboard {
    /// Update the batch gauge mid-epoch.
    pub fn batch_tick(&mut self, epoch: usize, batch: usize) -> Result<(), String> {
        self.state.epoch = epoch;
        self.state.batch = batch;
        self.terminal
            .draw(|f| draw(f, &self.state))
            .map_err(|e| e.to_string())?;
        self.drain_quit()
    }

    /// Record a finished epoch.
    pub fn epoch_done(&mut self, metrics: EpochMetrics) -> Result<(), String> {
        self.state.epoch += 1;
        self.state.batch = 0;
        self.state.history.push(metrics);
        self.terminal
            .draw(|f| draw(f, &self.state))
            .map_err(|e| e.to_string())?;
        self.drain_quit()
    }

    /// Mark the run complete and wait for `q`.
    pub fn finish(mut self) -> Result<(), String> {
        self.state.done = true;
        self.terminal
            .draw(|f| draw(f, &self.state))
            .map_err(|e| e.to_string())?;
        wait_for_quit(&mut self.terminal, &self.state)?;
        ratatui::restore();
        Ok(())
    }

    fn drain_quit(&mut self) -> Result<(), String> {
        while event::poll(std::time::Duration::ZERO).map_err(|e| e.to_string())? {
            if is_quit(&event::read().map_err(|e| e.to_string())?) {
                ratatui::restore();
                std::process::exit(0);
            }
        }
        Ok(())
    }
}

/// Render a recorded run: every epoch frame in order, no clocks, then the
/// completed frame until `q`.
pub fn run_replay(replay: &Replay) -> Result<(), String> {
    let mut state = DashState::new(
        &replay.device,
        &replay.model,
        replay.epochs.len(),
        replay.batches_per_epoch,
        replay.memory.clone(),
    );
    let mut terminal = ratatui::init();
    for m in &replay.epochs {
        state.epoch += 1;
        state.batch = replay.batches_per_epoch;
        state.history.push(m.clone());
        terminal
            .draw(|f| draw(f, &state))
            .map_err(|e| e.to_string())?;
    }
    state.done = true;
    terminal
        .draw(|f| draw(f, &state))
        .map_err(|e| e.to_string())?;
    wait_for_quit(&mut terminal, &state)?;
    ratatui::restore();
    Ok(())
}

fn wait_for_quit(terminal: &mut DefaultTerminal, state: &DashState) -> Result<(), String> {
    loop {
        let ev = event::read().map_err(|e| e.to_string())?;
        if is_quit(&ev) {
            return Ok(());
        }
        if matches!(ev, Event::Resize(..)) {
            terminal
                .draw(|f| draw(f, state))
                .map_err(|e| e.to_string())?;
        }
    }
}

fn is_quit(ev: &Event) -> bool {
    matches!(
        ev,
        Event::Key(k) if k.code == KeyCode::Char('q') || k.code == KeyCode::Esc
    )
}

fn draw(frame: &mut Frame<'_>, s: &DashState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(8),    // loss
            Constraint::Min(8),    // accuracy
            Constraint::Length(3), // epoch gauge
            Constraint::Length(3), // batch gauge
            Constraint::Length(4), // stats
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

    header(frame, rows[0], s);
    spark(
        frame,
        rows[1],
        "loss",
        s.history.iter().map(|m| m.loss).collect(),
        s.latest().map(|m| format!("{:.4}", m.loss)),
        Color::LightRed,
        true,
    );
    spark(
        frame,
        rows[2],
        "accuracy",
        s.history.iter().map(|m| m.accuracy).collect(),
        s.latest().map(|m| format!("{:.1}%", m.accuracy * 100.0)),
        Color::LightGreen,
        false,
    );
    gauge(frame, rows[3], "epoch", s.epoch, s.epochs_total);
    gauge(frame, rows[4], "batch", s.batch, s.batches_total);
    stats(frame, rows[5], s);
    let hint = if s.done {
        " training complete — press q to exit "
    } else {
        " q to quit "
    };
    frame.render_widget(
        Paragraph::new(hint).style(Style::default().add_modifier(Modifier::REVERSED)),
        rows[6],
    );
}

fn header(frame: &mut Frame<'_>, area: Rect, s: &DashState) {
    let title = Line::from(vec![
        Span::styled(
            " oxmera train ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("— {} — device {} ", s.model, s.device)),
    ]);
    frame.render_widget(
        Paragraph::new(title).block(Block::default().borders(Borders::ALL)),
        area,
    );
}

/// Sparkline over per-epoch values. Losses are inverted (smaller is
/// taller reads wrong) — no: losses render as-is; `falling` only affects
/// the value scale so early large losses don't flatten the tail.
fn spark(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    values: Vec<f32>,
    current: Option<String>,
    color: Color,
    falling: bool,
) {
    let scaled: Vec<u64> = if values.is_empty() {
        Vec::new()
    } else {
        let max = values.iter().cloned().fold(f32::MIN, f32::max).max(1e-9);
        let min = values.iter().cloned().fold(f32::MAX, f32::min);
        values
            .iter()
            .map(|&v| {
                let norm = if falling && max > min {
                    (v - min) / (max - min)
                } else {
                    v / max
                };
                (norm.clamp(0.0, 1.0) * 100.0) as u64 + 1
            })
            .collect()
    };
    let label = match current {
        Some(c) => format!(" {title} — {c} "),
        None => format!(" {title} "),
    };
    frame.render_widget(
        Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title(label))
            .style(Style::default().fg(color))
            .data(&scaled),
        area,
    );
}

fn gauge(frame: &mut Frame<'_>, area: Rect, title: &str, done: usize, total: usize) {
    let ratio = if total == 0 {
        0.0
    } else {
        (done as f64 / total as f64).clamp(0.0, 1.0)
    };
    frame.render_widget(
        Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {title} ")),
            )
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio)
            .label(format!("{done}/{total}")),
        area,
    );
}

fn stats(frame: &mut Frame<'_>, area: Rect, s: &DashState) {
    let throughput = s
        .latest()
        .map(|m| format!("{:.0} samples/s", m.throughput))
        .unwrap_or_else(|| "-".into());
    let memory = if s.memory.is_empty() {
        "-".into()
    } else {
        s.memory.clone()
    };
    let text = vec![
        Line::from(format!(" throughput   {throughput}")),
        Line::from(format!(" memory       {memory}")),
    ];
    frame.render_widget(
        Paragraph::new(text).block(Block::default().borders(Borders::ALL).title(" stats ")),
        area,
    );
}
