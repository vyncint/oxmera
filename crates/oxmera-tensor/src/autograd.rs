//! Reverse-mode autograd plumbing: the tape nodes tensors carry, gradient
//! accumulation, and the recording switch.
//!
//! The differentiable *rules* (VJPs) live next to the ops that create them
//! (`crate::ops`); this module owns the graph mechanics only.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxmera_core::Result;

use crate::tensor::Tensor;

/// One recorded operation: the inputs it consumed and the vector-Jacobian
/// product mapping the output gradient to input gradients (aligned with
/// `inputs`; `None` for non-differentiable inputs such as indices).
pub struct GradFn {
    /// The tensors the op consumed, in order.
    pub inputs: Vec<Tensor>,
    /// The VJP: output gradient in, one optional gradient per input out.
    #[allow(clippy::type_complexity)]
    pub vjp: Box<dyn Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync>,
}

impl std::fmt::Debug for GradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradFn")
            .field("inputs", &self.inputs.len())
            .finish()
    }
}

/// The autograd state a tracked tensor carries.
#[derive(Debug)]
pub struct AutogradMeta {
    /// Whether gradients accumulate on this tensor during `backward`.
    pub(crate) requires_grad: bool,
    /// The accumulated gradient, if any backward pass has reached it.
    pub(crate) grad: Mutex<Option<Tensor>>,
    /// How this tensor was computed; `None` for leaves.
    pub(crate) grad_fn: Option<GradFn>,
}

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(true) };
}

/// Whether ops on this thread currently record onto the tape.
pub fn is_recording() -> bool {
    RECORDING.with(Cell::get)
}

/// Run `f` with tape recording disabled — the inference/`no_grad` context.
///
/// Nested calls are fine; recording resumes when the outermost guard ends.
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = NoGradGuard::new();
    f()
}

/// RAII guard that disables tape recording until dropped.
pub struct NoGradGuard {
    previous: bool,
}

impl NoGradGuard {
    /// Disable recording on this thread until the guard drops.
    pub fn new() -> Self {
        let previous = RECORDING.with(|r| r.replace(false));
        Self { previous }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        RECORDING.with(|r| r.set(previous));
    }
}

fn meta_id(meta: &Arc<AutogradMeta>) -> usize {
    Arc::as_ptr(meta) as usize
}

/// Reverse-topological gradient propagation from `root`, seeding with
/// `seed`. Called by [`Tensor::backward`].
pub(crate) fn run_backward(root: &Tensor, seed: Tensor) -> Result<()> {
    // Everything backward computes is bookkeeping, not new graph.
    no_grad(|| run_backward_inner(root, seed))
}

fn run_backward_inner(root: &Tensor, seed: Tensor) -> Result<()> {
    let Some(root_meta) = root.autograd_meta() else {
        return Ok(());
    };

    // Post-order DFS for a topological order over the tape.
    let mut order: Vec<Tensor> = Vec::new();
    let mut visited: HashMap<usize, ()> = HashMap::new();
    let mut stack: Vec<(Tensor, bool)> = vec![(root.clone(), false)];
    while let Some((t, expanded)) = stack.pop() {
        let Some(meta) = t.autograd_meta() else {
            continue;
        };
        let id = meta_id(&meta);
        if expanded {
            order.push(t);
            continue;
        }
        if visited.contains_key(&id) {
            continue;
        }
        visited.insert(id, ());
        stack.push((t.clone(), true));
        if let Some(gf) = &meta.grad_fn {
            for input in &gf.inputs {
                stack.push((input.clone(), false));
            }
        }
    }

    let mut pending: HashMap<usize, Tensor> = HashMap::new();
    pending.insert(meta_id(&root_meta), seed);

    for t in order.into_iter().rev() {
        let meta = t.autograd_meta().expect("ordered tensors carry meta");
        let id = meta_id(&meta);
        let Some(grad) = pending.remove(&id) else {
            continue;
        };

        if meta.requires_grad {
            let mut slot = meta.grad.lock().expect("grad mutex poisoned");
            *slot = Some(match slot.take() {
                Some(existing) => existing.add(&grad)?,
                None => grad.clone(),
            });
        }

        if let Some(gf) = &meta.grad_fn {
            let input_grads = (gf.vjp)(&grad)?;
            debug_assert_eq!(input_grads.len(), gf.inputs.len());
            for (input, ig) in gf.inputs.iter().zip(input_grads) {
                let (Some(im), Some(ig)) = (input.autograd_meta(), ig) else {
                    continue;
                };
                let iid = meta_id(&im);
                let accumulated = match pending.remove(&iid) {
                    Some(existing) => existing.add(&ig)?,
                    None => ig,
                };
                pending.insert(iid, accumulated);
            }
        }
    }
    Ok(())
}
