//! Strided offset iteration: walk a layout's storage offsets in logical
//! (row-major) order, starting from any linear position — the primitive
//! that lets rayon chunks start mid-tensor.

use oxmera_core::Layout;

/// An odometer over a layout, yielding storage offsets in logical order.
pub struct OffsetWalker {
    dims: Vec<usize>,
    strides: Vec<isize>,
    coords: Vec<usize>,
    offset: isize,
}

impl OffsetWalker {
    /// A walker positioned at linear index `start`.
    pub fn at(layout: &Layout, start: usize) -> Self {
        let dims = layout.shape.dims().to_vec();
        let strides = layout.strides.values().to_vec();
        let mut coords = vec![0usize; dims.len()];
        let mut offset = layout.offset as isize;
        let mut rem = start;
        for i in (0..dims.len()).rev() {
            if dims[i] == 0 {
                continue;
            }
            let c = rem % dims[i];
            rem /= dims[i];
            coords[i] = c;
            offset += c as isize * strides[i];
        }
        Self {
            dims,
            strides,
            coords,
            offset,
        }
    }

    /// The current storage offset; advances the walker by one element.
    pub fn next_offset(&mut self) -> usize {
        let current = self.offset as usize;
        let mut d = self.dims.len();
        loop {
            if d == 0 {
                break;
            }
            d -= 1;
            self.coords[d] += 1;
            self.offset += self.strides[d];
            if self.coords[d] < self.dims[d] {
                break;
            }
            self.offset -= self.dims[d] as isize * self.strides[d];
            self.coords[d] = 0;
        }
        current
    }
}
