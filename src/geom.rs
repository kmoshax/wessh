//! 2D similarity/affine helpers used for face alignment.

pub type Point = [f32; 2];
pub type Landmarks5 = [Point; 5];

/// Row-major 2x3 affine matrix `[a, b, tx, c, d, ty]` mapping `(x, y) -> (a*x + b*y + tx, c*x + d*y + ty)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f32; 6]);

impl Affine {
    pub fn apply(&self, p: Point) -> Point {
        let m = &self.0;
        [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]]
    }

    pub fn invert(&self) -> Affine {
        let [a, b, tx, c, d, ty] = self.0;
        let det = a * d - b * c;
        let inv = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
        let (ia, ib, ic, id) = (d * inv, -b * inv, -c * inv, a * inv);
        Affine([ia, ib, -(ia * tx + ib * ty), ic, id, -(ic * tx + id * ty)])
    }

    /// Least-squares similarity transform (rotation + uniform scale + translation) from `src` to `dst`.
    /// Equivalent to OpenCV's `estimateAffinePartial2D` without outlier rejection.
    pub fn similarity(src: &[Point], dst: &[Point]) -> Affine {
        let n = src.len() as f32;
        let (mut msx, mut msy, mut mdx, mut mdy) = (0.0, 0.0, 0.0, 0.0);
        for (s, d) in src.iter().zip(dst) {
            msx += s[0];
            msy += s[1];
            mdx += d[0];
            mdy += d[1];
        }
        msx /= n;
        msy /= n;
        mdx /= n;
        mdy /= n;
        let (mut num_a, mut num_b, mut den) = (0.0f32, 0.0f32, 0.0f32);
        for (s, d) in src.iter().zip(dst) {
            let (xs, ys) = (s[0] - msx, s[1] - msy);
            let (xd, yd) = (d[0] - mdx, d[1] - mdy);
            num_a += xs * xd + ys * yd;
            num_b += xs * yd - ys * xd;
            den += xs * xs + ys * ys;
        }
        let den = den.max(1e-12);
        let (a, b) = (num_a / den, num_b / den);
        let tx = mdx - (a * msx - b * msy);
        let ty = mdy - (b * msx + a * msy);
        Affine([a, -b, tx, b, a, ty])
    }

    /// Uniform scale factor of the transform.
    pub fn scale(&self) -> f32 {
        (self.0[0] * self.0[0] + self.0[3] * self.0[3]).sqrt()
    }
}

/// Normalized 5-point alignment templates (from InsightFace / FaceFusion).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    Arcface112V2,
    Arcface128,
}

impl Template {
    pub fn points(self) -> Landmarks5 {
        match self {
            Template::Arcface112V2 => [
                [0.34191607, 0.46157411],
                [0.65653393, 0.45983393],
                [0.50022500, 0.64050536],
                [0.37097589, 0.82469196],
                [0.63151696, 0.82325089],
            ],
            Template::Arcface128 => [
                [0.36167656, 0.40387734],
                [0.63696719, 0.40235469],
                [0.50019687, 0.56044219],
                [0.38710391, 0.72160547],
                [0.61507734, 0.72034453],
            ],
        }
    }

    /// Transform mapping frame coordinates onto a `size`x`size` aligned crop.
    pub fn align(self, landmarks: &Landmarks5, size: usize) -> Affine {
        let dst = self.points().map(|p| [p[0] * size as f32, p[1] * size as f32]);
        Affine::similarity(landmarks, &dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_recovers_known_transform() {
        let t = Affine([1.5, -0.4, 10.0, 0.4, 1.5, -3.0]);
        let src: Vec<Point> = vec![[0.0, 0.0], [10.0, 2.0], [3.0, 7.0], [8.0, 9.0], [1.0, 5.0]];
        let dst: Vec<Point> = src.iter().map(|&p| t.apply(p)).collect();
        let est = Affine::similarity(&src, &dst);
        for (a, b) in est.0.iter().zip(t.0.iter()) {
            assert!((a - b).abs() < 1e-3, "{est:?} vs {t:?}");
        }
        let back = est.invert().apply(est.apply([4.0, 4.0]));
        assert!((back[0] - 4.0).abs() < 1e-3 && (back[1] - 4.0).abs() < 1e-3);
    }
}
