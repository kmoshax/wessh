//! Temporal stabilisation: One Euro filtering of landmarks and frame-to-frame face association.

use crate::geom::Landmarks5;

/// One Euro filter (Casiez et al.): strong smoothing when still, low lag when moving.
#[derive(Clone)]
struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    x: Option<f32>,
    dx: f32,
}

fn alpha(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl OneEuro {
    fn new(min_cutoff: f32, beta: f32) -> Self {
        Self { min_cutoff, beta, x: None, dx: 0.0 }
    }

    fn filter(&mut self, value: f32, dt: f32) -> f32 {
        let Some(prev) = self.x else {
            self.x = Some(value);
            return value;
        };
        let dx = (value - prev) / dt;
        let a_d = alpha(1.0, dt);
        self.dx = a_d * dx + (1.0 - a_d) * self.dx;
        let cutoff = self.min_cutoff + self.beta * self.dx.abs();
        let a = alpha(cutoff, dt);
        let x = a * value + (1.0 - a) * prev;
        self.x = Some(x);
        x
    }
}

struct Track {
    filters: Vec<OneEuro>,
    center: [f32; 2],
    seen: bool,
}

/// Keeps one filter bank per visible face, matched by nearest centre between frames.
pub struct FaceSmoother {
    tracks: Vec<Track>,
    pub min_cutoff: f32,
    pub beta: f32,
}

impl FaceSmoother {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self { tracks: Vec::new(), min_cutoff, beta }
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Smooth landmarks in place. `face_size` values are used as association radii.
    pub fn smooth(&mut self, faces: &mut [(Landmarks5, f32)], dt: f32) {
        let dt = dt.clamp(1e-3, 0.5);
        self.tracks.iter_mut().for_each(|t| t.seen = false);
        for (lm, size) in faces.iter_mut() {
            let c = centroid(lm);
            let best = self
                .tracks
                .iter_mut()
                .filter(|t| !t.seen)
                .map(|t| {
                    let d = ((t.center[0] - c[0]).powi(2) + (t.center[1] - c[1]).powi(2)).sqrt();
                    (d, t)
                })
                .filter(|(d, _)| *d < *size * 0.6)
                .min_by(|a, b| a.0.total_cmp(&b.0));
            let track = match best {
                Some((_, t)) => t,
                None => {
                    self.tracks.push(Track {
                        filters: vec![OneEuro::new(self.min_cutoff, self.beta); 10],
                        center: c,
                        seen: false,
                    });
                    self.tracks.last_mut().unwrap()
                }
            };
            track.seen = true;
            track.center = c;
            // Scale the speed coefficient by face size so behaviour is resolution independent.
            let norm = size.max(1.0);
            for (i, p) in lm.iter_mut().enumerate() {
                for k in 0..2 {
                    let f = &mut track.filters[i * 2 + k];
                    p[k] = f.filter(p[k] / norm, dt) * norm;
                }
            }
        }
        self.tracks.retain(|t| t.seen);
    }
}

fn centroid(lm: &Landmarks5) -> [f32; 2] {
    let (sx, sy) = lm.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
    [sx / 5.0, sy / 5.0]
}
