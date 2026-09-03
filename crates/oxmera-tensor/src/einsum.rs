//! Einstein summation for one or two operands, lowered onto the ops the
//! backends already implement: `permute`, `reshape`, `sum` and the
//! rank-3 `matmul`. Every step is a recorded op, so `einsum` is
//! differentiable without a VJP of its own.
//!
//! The spec is the NumPy form with an explicit arrow — `"ij,jk->ik"`,
//! `"rbhd,rdo->rbho"`, `"ij->ji"`, `"i,i->"` — over lowercase ASCII
//! letters. Each letter names one dimension size; a letter that appears in
//! both operands and not in the output is contracted, one that appears in
//! both and in the output is a batch dimension, one that appears in a
//! single operand and not in the output is summed. Repeated letters inside
//! one operand (the diagonal, `"ii->i"`) and implicit output are not
//! supported and report a typed error.

use oxmera_core::{Error, Result, Shape};

use crate::tensor::Tensor;

/// One parsed operand: its letters, in dimension order.
type Subscript = Vec<char>;

fn invalid(detail: String) -> Error {
    Error::InvalidArgument {
        op: "einsum",
        detail,
    }
}

fn parse(spec: &str, operands: usize) -> Result<(Vec<Subscript>, Subscript)> {
    let compact: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = compact.split_once("->").ok_or_else(|| {
        invalid(format!(
            "spec {spec:?} needs an explicit output after '->' (implicit output is not supported)"
        ))
    })?;
    let inputs: Vec<Subscript> = lhs.split(',').map(|s| s.chars().collect()).collect();
    if inputs.len() != operands {
        return Err(invalid(format!(
            "spec {spec:?} names {} operand(s) but {operands} were given",
            inputs.len()
        )));
    }
    let output: Subscript = rhs.chars().collect();
    for letters in inputs.iter().chain(std::iter::once(&output)) {
        for &c in letters {
            if !c.is_ascii_lowercase() {
                return Err(invalid(format!(
                    "spec {spec:?}: subscripts are lowercase ASCII letters, got {c:?}"
                )));
            }
        }
        let mut sorted = letters.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != letters.len() {
            return Err(invalid(format!(
                "spec {spec:?}: a letter repeats within one subscript (diagonals are not supported)"
            )));
        }
    }
    let all_inputs: Vec<char> = inputs.iter().flatten().copied().collect();
    for &c in &output {
        if !all_inputs.contains(&c) {
            return Err(invalid(format!(
                "spec {spec:?}: output letter {c:?} does not appear in any operand"
            )));
        }
    }
    Ok((inputs, output))
}

/// Record every letter's size, requiring equal sizes wherever a letter
/// appears twice.
fn sizes(inputs: &[Subscript], operands: &[&Tensor], spec: &str) -> Result<Vec<(char, usize)>> {
    let mut table: Vec<(char, usize)> = Vec::new();
    for (sub, t) in inputs.iter().zip(operands) {
        if sub.len() != t.ndim() {
            return Err(invalid(format!(
                "spec {spec:?}: subscript {} has {} letters but the operand has rank {}",
                sub.iter().collect::<String>(),
                sub.len(),
                t.ndim()
            )));
        }
        for (&c, &d) in sub.iter().zip(t.dims()) {
            match table.iter().find(|(k, _)| *k == c) {
                Some(&(_, existing)) if existing != d => {
                    return Err(invalid(format!(
                        "spec {spec:?}: letter {c:?} is {existing} in one operand and {d} in another"
                    )));
                }
                Some(_) => {}
                None => table.push((c, d)),
            }
        }
    }
    Ok(table)
}

fn size_of(table: &[(char, usize)], c: char) -> usize {
    table
        .iter()
        .find(|(k, _)| *k == c)
        .map(|(_, d)| *d)
        .unwrap_or(1)
}

/// Permute `t` so its letters appear in `order`, then sum away every
/// letter not in `keep` (they must come last in `order`). Returns the
/// tensor with letters exactly `keep`, in that order.
fn arrange(t: &Tensor, sub: &Subscript, order: &[char]) -> Result<Tensor> {
    let perm: Vec<usize> = order
        .iter()
        .map(|c| sub.iter().position(|k| k == c).expect("letter present"))
        .collect();
    let identity: Vec<usize> = (0..perm.len()).collect();
    if perm == identity {
        Ok(t.clone())
    } else {
        t.permute(&perm)
    }
}

fn sum_letters(t: &Tensor, letters: &[char], drop: &[char]) -> Result<(Tensor, Vec<char>)> {
    let axes: Vec<usize> = letters
        .iter()
        .enumerate()
        .filter(|(_, c)| drop.contains(c))
        .map(|(i, _)| i)
        .collect();
    let kept: Vec<char> = letters
        .iter()
        .copied()
        .filter(|c| !drop.contains(c))
        .collect();
    if axes.is_empty() {
        Ok((t.clone(), kept))
    } else {
        Ok((t.sum(&axes)?, kept))
    }
}

/// Einstein summation over one or two operands. See the module docs for
/// the accepted specs.
///
/// ```
/// use oxmera_tensor::{Tensor, einsum};
///
/// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
/// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();
/// let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
/// assert_eq!(c.to_vec_f32().unwrap(), a.matmul(&b).unwrap().to_vec_f32().unwrap());
/// let t = einsum("ij->ji", &[&a]).unwrap();
/// assert_eq!(t.to_vec_f32().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
/// ```
pub fn einsum(spec: &str, operands: &[&Tensor]) -> Result<Tensor> {
    if operands.is_empty() || operands.len() > 2 {
        return Err(invalid(format!(
            "einsum takes one or two operands, got {}",
            operands.len()
        )));
    }
    let (inputs, output) = parse(spec, operands.len())?;
    let table = sizes(&inputs, operands, spec)?;

    if operands.len() == 1 {
        let (sub, t) = (&inputs[0], operands[0]);
        // Output letters first (in output order), summed letters last.
        let dropped: Vec<char> = sub
            .iter()
            .copied()
            .filter(|c| !output.contains(c))
            .collect();
        let order: Vec<char> = output
            .iter()
            .copied()
            .chain(dropped.iter().copied())
            .collect();
        let arranged = arrange(t, sub, &order)?;
        let (summed, _) = sum_letters(&arranged, &order, &dropped)?;
        return Ok(summed);
    }

    let (sa, sb) = (&inputs[0], &inputs[1]);
    let (a, b) = (operands[0], operands[1]);
    let in_both = |c: &char| sa.contains(c) && sb.contains(c);
    // Letters only one operand has and the output does not: sum them away
    // before the product so they never enter the matmul.
    let a_drop: Vec<char> = sa
        .iter()
        .copied()
        .filter(|c| !sb.contains(c) && !output.contains(c))
        .collect();
    let b_drop: Vec<char> = sb
        .iter()
        .copied()
        .filter(|c| !sa.contains(c) && !output.contains(c))
        .collect();
    let batch: Vec<char> = output.iter().copied().filter(in_both).collect();
    let contract: Vec<char> = sa
        .iter()
        .copied()
        .filter(|c| sb.contains(c) && !output.contains(c))
        .collect();
    let a_free: Vec<char> = output
        .iter()
        .copied()
        .filter(|c| sa.contains(c) && !sb.contains(c))
        .collect();
    let b_free: Vec<char> = output
        .iter()
        .copied()
        .filter(|c| sb.contains(c) && !sa.contains(c))
        .collect();

    // A → [batch.., a_free.., contract.., a_drop..] then sum a_drop.
    let a_order: Vec<char> = batch
        .iter()
        .chain(&a_free)
        .chain(&contract)
        .chain(&a_drop)
        .copied()
        .collect();
    let (a_arr, a_letters) = sum_letters(&arrange(a, sa, &a_order)?, &a_order, &a_drop)?;
    // B → [batch.., contract.., b_free.., b_drop..] then sum b_drop.
    let b_order: Vec<char> = batch
        .iter()
        .chain(&contract)
        .chain(&b_free)
        .chain(&b_drop)
        .copied()
        .collect();
    let (b_arr, _) = sum_letters(&arrange(b, sb, &b_order)?, &b_order, &b_drop)?;
    debug_assert_eq!(a_letters.len(), batch.len() + a_free.len() + contract.len());

    let dim = |letters: &[char]| -> usize { letters.iter().map(|&c| size_of(&table, c)).product() };
    let (bn, m, k, n) = (dim(&batch), dim(&a_free), dim(&contract), dim(&b_free));
    let a3 = a_arr.contiguous()?.reshape(Shape::from([bn, m, k]))?;
    let b3 = b_arr.contiguous()?.reshape(Shape::from([bn, k, n]))?;
    let prod = a3.matmul(&b3)?; // [bn, m, n]

    // Unflatten to [batch.., a_free.., b_free..] and permute into output order.
    let natural: Vec<char> = batch
        .iter()
        .chain(&a_free)
        .chain(&b_free)
        .copied()
        .collect();
    let natural_dims: Vec<usize> = natural.iter().map(|&c| size_of(&table, c)).collect();
    let unflat = prod.reshape(Shape::new(natural_dims))?;
    arrange(&unflat, &natural, &output)
}
