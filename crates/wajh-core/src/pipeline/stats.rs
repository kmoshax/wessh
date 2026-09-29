use std::ops::{AddAssign, Div};

/// Per-frame timings in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub faces: usize,
    pub detect_ms: f32,
    pub swap_ms: f32,
    pub enhance_ms: f32,
    /// Occlusion / region / mouth masks and colour correction (part of `swap_ms`).
    pub mask_ms: f32,
    pub total_ms: f32,
}

impl AddAssign for Stats {
    fn add_assign(&mut self, rhs: Self) {
        self.faces += rhs.faces;
        self.detect_ms += rhs.detect_ms;
        self.swap_ms += rhs.swap_ms;
        self.enhance_ms += rhs.enhance_ms;
        self.mask_ms += rhs.mask_ms;
        self.total_ms += rhs.total_ms;
    }
}

impl Div<f32> for Stats {
    type Output = Self;

    /// Average over `n` frames.
    fn div(self, n: f32) -> Self {
        Self {
            faces: (self.faces as f32 / n).round() as usize,
            detect_ms: self.detect_ms / n,
            swap_ms: self.swap_ms / n,
            enhance_ms: self.enhance_ms / n,
            mask_ms: self.mask_ms / n,
            total_ms: self.total_ms / n,
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "detect {:.2} ms | swap {:.2} ms (masks {:.2}) | enhance {:.2} ms | total {:.2} ms ({:.1} fps)",
            self.detect_ms,
            self.swap_ms,
            self.mask_ms,
            self.enhance_ms,
            self.total_ms,
            1000.0 / self.total_ms.max(f32::EPSILON)
        )
    }
}
