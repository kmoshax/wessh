use std::ops::{AddAssign, Div};

/// Per-frame timings in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub faces: usize,
    /// Faces that reused an earlier frame's swap (see `Settings::max_faces_per_frame`).
    pub swaps_reused: usize,
    pub detect_ms: f32,
    pub swap_ms: f32,
    pub enhance_ms: f32,
    /// Occlusion / region / mouth masks and colour correction (part of `swap_ms`).
    pub mask_ms: f32,
    /// Occluder inference and its mask (part of `mask_ms`).
    pub occlusion_ms: f32,
    /// Parser inference and the region / mouth masks (part of `mask_ms`).
    pub parser_ms: f32,
    /// Model masks computed this frame.
    pub masks_computed: usize,
    /// Model masks reused from an earlier frame (live mode).
    pub masks_reused: usize,
    pub total_ms: f32,
}

impl Stats {
    /// Share of model masks served from the live-mode cache, if any were needed.
    #[must_use]
    pub fn mask_hit_rate(&self) -> Option<f32> {
        let total = self.masks_computed + self.masks_reused;
        (total > 0).then(|| self.masks_reused as f32 / total as f32)
    }
}

impl AddAssign for Stats {
    fn add_assign(&mut self, rhs: Self) {
        self.faces += rhs.faces;
        self.swaps_reused += rhs.swaps_reused;
        self.detect_ms += rhs.detect_ms;
        self.swap_ms += rhs.swap_ms;
        self.enhance_ms += rhs.enhance_ms;
        self.mask_ms += rhs.mask_ms;
        self.occlusion_ms += rhs.occlusion_ms;
        self.parser_ms += rhs.parser_ms;
        self.masks_computed += rhs.masks_computed;
        self.masks_reused += rhs.masks_reused;
        self.total_ms += rhs.total_ms;
    }
}

impl Div<f32> for Stats {
    type Output = Self;

    /// Average over `n` frames. Mask counts stay totals, so the hit rate is kept exact.
    fn div(self, n: f32) -> Self {
        Self {
            faces: (self.faces as f32 / n).round() as usize,
            swaps_reused: (self.swaps_reused as f32 / n).round() as usize,
            detect_ms: self.detect_ms / n,
            swap_ms: self.swap_ms / n,
            enhance_ms: self.enhance_ms / n,
            mask_ms: self.mask_ms / n,
            occlusion_ms: self.occlusion_ms / n,
            parser_ms: self.parser_ms / n,
            masks_computed: self.masks_computed,
            masks_reused: self.masks_reused,
            total_ms: self.total_ms / n,
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "detect {:.2} ms | swap {:.2} ms (", self.detect_ms, self.swap_ms)?;
        if self.swaps_reused > 0 {
            write!(f, "{} of {} faces reused, ", self.swaps_reused, self.faces)?;
        }
        write!(f, "masks {:.2}", self.mask_ms)?;
        if self.occlusion_ms > 0.0 || self.parser_ms > 0.0 {
            write!(f, ": occlusion {:.2}, parser {:.2}", self.occlusion_ms, self.parser_ms)?;
        }
        if let Some(rate) = self.mask_hit_rate() {
            write!(f, ", {:.0}% reused", rate * 100.0)?;
        }
        write!(
            f,
            ") | enhance {:.2} ms | total {:.2} ms ({:.1} fps)",
            self.enhance_ms,
            self.total_ms,
            1000.0 / self.total_ms.max(f32::EPSILON)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Stats;

    #[test]
    fn averaging_keeps_the_hit_rate() {
        let frame = Stats { masks_computed: 1, masks_reused: 3, mask_ms: 2.0, ..Stats::default() };
        let mut sum = Stats::default();
        for _ in 0..4 {
            sum += frame;
        }
        let mean = sum / 4.0;
        assert_eq!(mean.mask_ms, 2.0);
        assert_eq!(mean.mask_hit_rate(), Some(0.75));
        assert_eq!(Stats::default().mask_hit_rate(), None);
    }
}
