//! Temporal stabilisation: One Euro filtering of landmarks, with faces associated across frames.

use crate::face::Face;

/// One Euro filter (Casiez et al. 2012): heavy smoothing when still, little lag when moving.
#[derive(Clone, Debug)]
struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    value: Option<f32>,
    derivative: f32,
}

impl OneEuro {
    const DERIVATIVE_CUTOFF: f32 = 1.0;

    const fn new(min_cutoff: f32, beta: f32) -> Self {
        Self { min_cutoff, beta, value: None, derivative: 0.0 }
    }

    fn alpha(cutoff: f32, dt: f32) -> f32 {
        let tau = 1.0 / (std::f32::consts::TAU * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    fn filter(&mut self, x: f32, dt: f32) -> f32 {
        let Some(previous) = self.value else {
            self.value = Some(x);
            return x;
        };
        let a_d = Self::alpha(Self::DERIVATIVE_CUTOFF, dt);
        self.derivative = a_d.mul_add((x - previous) / dt, (1.0 - a_d) * self.derivative);
        let a = Self::alpha(self.beta.mul_add(self.derivative.abs(), self.min_cutoff), dt);
        let filtered = a.mul_add(x, (1.0 - a) * previous);
        self.value = Some(filtered);
        filtered
    }
}

#[derive(Debug)]
struct Track {
    /// x and y for each of the 5 landmarks.
    filters: [OneEuro; 10],
    center: [f32; 2],
    matched: bool,
}

/// One filter bank per visible face, matched to detections by nearest centre.
#[derive(Debug)]
pub struct FaceSmoother {
    tracks: Vec<Track>,
    /// Cutoff (Hz) when still; lower = smoother.
    pub min_cutoff: f32,
    /// Speed coefficient, in face-sizes per second; higher = less lag.
    pub beta: f32,
}

impl Default for FaceSmoother {
    fn default() -> Self {
        Self { tracks: Vec::new(), min_cutoff: 1.2, beta: 8.0 }
    }
}

impl FaceSmoother {
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Smooth landmarks in place; `dt` is seconds since the previous frame.
    pub fn smooth(&mut self, faces: &mut [Face], dt: f32) {
        let dt = dt.clamp(1e-3, 0.5);
        self.tracks.iter_mut().for_each(|t| t.matched = false);
        for face in faces {
            let scale = face.size().max(1.0);
            let track = self.track_for(centroid(face), scale);
            // Filter in face-size units so behaviour doesn't depend on resolution or distance.
            for (point, filters) in face.landmarks.iter_mut().zip(track.filters.chunks_exact_mut(2)) {
                for (coordinate, filter) in point.iter_mut().zip(filters) {
                    *coordinate = filter.filter(*coordinate / scale, dt) * scale;
                }
            }
        }
        self.tracks.retain(|t| t.matched);
    }

    fn track_for(&mut self, center: [f32; 2], scale: f32) -> &mut Track {
        let nearest = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.matched)
            .map(|(i, t)| (i, (t.center[0] - center[0]).hypot(t.center[1] - center[1])))
            .filter(|&(_, distance)| distance < scale * 0.6)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
        let index = nearest.unwrap_or_else(|| {
            self.tracks.push(Track { filters: std::array::from_fn(|_| OneEuro::new(self.min_cutoff, self.beta)), center, matched: false });
            self.tracks.len() - 1
        });
        let track = &mut self.tracks[index];
        track.center = center;
        track.matched = true;
        track
    }
}

fn centroid(face: &Face) -> [f32; 2] {
    let (x, y) = face.landmarks.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
    [x / 5.0, y / 5.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_at(x: f32) -> Face {
        Face { bbox: [x - 50.0, 0.0, x + 50.0, 100.0], score: 1.0, landmarks: [[x, 50.0]; 5] }
    }

    #[test]
    fn jitter_is_damped() {
        let mut smoother = FaceSmoother::default();
        let mut last = 0.0;
        for i in 0..30 {
            let mut faces = [face_at(if i % 2 == 0 { 200.0 } else { 202.0 })];
            smoother.smooth(&mut faces, 1.0 / 30.0);
            last = faces[0].landmarks[0][0];
        }
        assert!((last - 201.0).abs() < 1.0, "{last}");
    }
}
