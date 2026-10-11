//! HCT (hue, chroma, tone) colour space from Material Color Utilities.
//!
//! Hue and chroma come from CAM16 under default viewing conditions; tone is CIE L*.
//! Solving for a colour with a requested hue/chroma/tone uses the reference
//! bisection search: find the CAM16 lightness `J` that yields the tone, and lower
//! the chroma until the result fits in sRGB.

use std::f64::consts::PI;
use std::sync::LazyLock;

/// A packed sRGB colour, `[r, g, b]`.
pub type Rgb = [u8; 3];

const WHITE_POINT_D65: [f64; 3] = [95.047, 100.0, 108.883];

const SRGB_TO_XYZ: [[f64; 3]; 3] = [[0.41233895, 0.35762064, 0.18051042], [0.2126, 0.7152, 0.0722], [0.01932141, 0.11916382, 0.95034478]];

const XYZ_TO_SRGB: [[f64; 3]; 3] = [
    [3.2413774792388685, -1.5376652402851851, -0.49885366846268053],
    [-0.9691452513005321, 1.8758853451067872, 0.04156585616912061],
    [0.05562093689691305, -0.20395524564742123, 1.0571799111220335],
];

const XYZ_TO_CAM16RGB: [[f64; 3]; 3] = [[0.401288, 0.650173, -0.051461], [-0.250268, 1.204414, 0.045854], [-0.002079, 0.048952, 0.953127]];

const CAM16RGB_TO_XYZ: [[f64; 3]; 3] =
    [[1.86206786, -1.01125463, 0.14918677], [0.38752654, 0.62144744, -0.00897398], [-0.01584150, -0.03412294, 1.04996444]];

/// CIE L*a*b* constants.
const LAB_EPSILON: f64 = 216.0 / 24389.0;
const LAB_KAPPA: f64 = 24389.0 / 27.0;

/// Bisection tolerances from the reference implementation.
const CHROMA_SEARCH_TOLERANCE: f64 = 0.4;
const J_SEARCH_TOLERANCE: f64 = 0.01;
const TONE_MATCH_TOLERANCE: f64 = 0.2;
const MAX_HUE_DRIFT: f64 = 1.0;

fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|row| m[row][0] * v[0] + m[row][1] * v[1] + m[row][2] * v[2])
}

fn sanitize_degrees(degrees: f64) -> f64 {
    degrees.rem_euclid(360.0)
}

/// sRGB channel (0..=255) → linear light (0..=100).
fn linearize(channel: u8) -> f64 {
    let normalized = f64::from(channel) / 255.0;
    let linear = if normalized <= 0.040449936 { normalized / 12.92 } else { ((normalized + 0.055) / 1.055).powf(2.4) };
    linear * 100.0
}

/// Linear light (0..=100) → sRGB channel, clamped.
fn delinearize(linear: f64) -> u8 {
    let normalized = linear / 100.0;
    let gamma = if normalized <= 0.0031308 { normalized * 12.92 } else { 1.055 * normalized.powf(1.0 / 2.4) - 0.055 };
    (gamma * 255.0).round().clamp(0.0, 255.0) as u8
}

fn xyz_from_rgb(rgb: Rgb) -> [f64; 3] {
    mul(&SRGB_TO_XYZ, rgb.map(linearize))
}

fn rgb_from_xyz(xyz: [f64; 3]) -> Rgb {
    mul(&XYZ_TO_SRGB, xyz).map(delinearize)
}

fn lab_f(t: f64) -> f64 {
    if t > LAB_EPSILON { t.cbrt() } else { (LAB_KAPPA * t + 16.0) / 116.0 }
}

fn lab_inv_f(ft: f64) -> f64 {
    let cubed = ft * ft * ft;
    if cubed > LAB_EPSILON { cubed } else { (116.0 * ft - 16.0) / LAB_KAPPA }
}

/// Relative luminance `Y` (0..=100) for a tone `L*`.
fn y_from_lstar(lstar: f64) -> f64 {
    100.0 * lab_inv_f((lstar + 16.0) / 116.0)
}

fn lstar_from_y(y: f64) -> f64 {
    116.0 * lab_f(y / 100.0) - 16.0
}

fn lstar_from_rgb(rgb: Rgb) -> f64 {
    lstar_from_y(xyz_from_rgb(rgb)[1])
}

fn gray_from_lstar(lstar: f64) -> Rgb {
    [delinearize(y_from_lstar(lstar)); 3]
}

/// CAM16 viewing conditions (the Material defaults: D65, grey world, average surround).
struct ViewingConditions {
    n: f64,
    aw: f64,
    nbb: f64,
    ncb: f64,
    c: f64,
    nc: f64,
    rgb_d: [f64; 3],
    fl: f64,
    fl_root: f64,
    z: f64,
}

impl ViewingConditions {
    fn standard() -> Self {
        const BACKGROUND_LSTAR: f64 = 50.0;
        const SURROUND: f64 = 2.0;
        let adapting_luminance = (200.0 / PI) * y_from_lstar(BACKGROUND_LSTAR) / 100.0;
        let rgb_w = mul(&XYZ_TO_CAM16RGB, WHITE_POINT_D65);
        let f = 0.8 + SURROUND / 10.0;
        let c = if f >= 0.9 { lerp(0.59, 0.69, (f - 0.9) * 10.0) } else { lerp(0.525, 0.59, (f - 0.8) * 10.0) };
        let d = (f * (1.0 - (1.0 / 3.6) * ((-adapting_luminance - 42.0) / 92.0).exp())).clamp(0.0, 1.0);
        let rgb_d = rgb_w.map(|w| d * (100.0 / w) + 1.0 - d);
        let k = 1.0 / (5.0 * adapting_luminance + 1.0);
        let k4 = k.powi(4);
        let k4f = 1.0 - k4;
        let fl = k4 * adapting_luminance + 0.1 * k4f * k4f * (5.0 * adapting_luminance).cbrt();
        let n = y_from_lstar(BACKGROUND_LSTAR) / WHITE_POINT_D65[1];
        let z = 1.48 + n.sqrt();
        let nbb = 0.725 / n.powf(0.2);
        let rgb_a = [0, 1, 2].map(|i| {
            let factor = (fl * rgb_d[i] * rgb_w[i] / 100.0).powf(0.42);
            400.0 * factor / (factor + 27.13)
        });
        let aw = (2.0 * rgb_a[0] + rgb_a[1] + 0.05 * rgb_a[2]) * nbb;
        Self { n, aw, nbb, ncb: nbb, c, nc: f, rgb_d, fl, fl_root: fl.powf(0.25), z }
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

static VIEWING: LazyLock<ViewingConditions> = LazyLock::new(ViewingConditions::standard);

/// A colour in CAM16, with the UCS coordinates used for colour distance.
#[derive(Clone, Copy, Debug)]
struct Cam16 {
    hue: f64,
    chroma: f64,
    j: f64,
    jstar: f64,
    astar: f64,
    bstar: f64,
}

impl Cam16 {
    fn from_rgb(rgb: Rgb) -> Self {
        let vc = &*VIEWING;
        let cam_rgb = mul(&XYZ_TO_CAM16RGB, xyz_from_rgb(rgb));
        let adapted = [0, 1, 2].map(|i| {
            let d = vc.rgb_d[i] * cam_rgb[i];
            let factor = (vc.fl * d.abs() / 100.0).powf(0.42);
            d.signum() * 400.0 * factor / (factor + 27.13)
        });
        let [r, g, b] = adapted;
        let a = (11.0 * r + -12.0 * g + b) / 11.0;
        let bb = (r + g - 2.0 * b) / 9.0;
        let u = (20.0 * r + 20.0 * g + 21.0 * b) / 20.0;
        let p2 = (40.0 * r + 20.0 * g + b) / 20.0;
        let hue = sanitize_degrees(bb.atan2(a).to_degrees());
        let ac = p2 * vc.nbb;
        let j = 100.0 * (ac / vc.aw).powf(vc.c * vc.z);
        let hue_prime = if hue < 20.14 { hue + 360.0 } else { hue };
        let e_hue = 0.25 * ((hue_prime.to_radians() + 2.0).cos() + 3.8);
        let p1 = 50000.0 / 13.0 * e_hue * vc.nc * vc.ncb;
        let t = p1 * a.hypot(bb) / (u + 0.305);
        let alpha = t.powf(0.9) * (1.64 - 0.29f64.powf(vc.n)).powf(0.73);
        let chroma = alpha * (j / 100.0).sqrt();
        Self::from_jch(j, chroma, hue)
    }

    fn from_jch(j: f64, chroma: f64, hue: f64) -> Self {
        let m = chroma * VIEWING.fl_root;
        let jstar = (1.0 + 100.0 * 0.007) * j / (1.0 + 0.007 * j);
        let mstar = 1.0 / 0.0228 * (1.0 + 0.0228 * m).ln();
        let radians = hue.to_radians();
        Self { hue, chroma, j, jstar, astar: mstar * radians.cos(), bstar: mstar * radians.sin() }
    }

    fn distance(&self, other: &Self) -> f64 {
        let d = (self.jstar - other.jstar).powi(2) + (self.astar - other.astar).powi(2) + (self.bstar - other.bstar).powi(2);
        1.41 * d.sqrt().powf(0.63)
    }

    fn to_rgb(self) -> Rgb {
        let vc = &*VIEWING;
        let alpha = if self.chroma == 0.0 || self.j == 0.0 { 0.0 } else { self.chroma / (self.j / 100.0).sqrt() };
        let t = (alpha / (1.64 - 0.29f64.powf(vc.n)).powf(0.73)).powf(1.0 / 0.9);
        let radians = self.hue.to_radians();
        let e_hue = 0.25 * ((radians + 2.0).cos() + 3.8);
        let ac = vc.aw * (self.j / 100.0).powf(1.0 / vc.c / vc.z);
        let p1 = e_hue * (50000.0 / 13.0) * vc.nc * vc.ncb;
        let p2 = ac / vc.nbb;
        let (sin, cos) = radians.sin_cos();
        let gamma = 23.0 * (p2 + 0.305) * t / (23.0 * p1 + 11.0 * t * cos + 108.0 * t * sin);
        let a = gamma * cos;
        let b = gamma * sin;
        let adapted = [
            (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0,
            (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0,
            (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0,
        ];
        let cam_rgb = [0, 1, 2].map(|i| {
            let value = adapted[i];
            let base = (27.13 * value.abs() / (400.0 - value.abs())).max(0.0);
            value.signum() * (100.0 / vc.fl) * base.powf(1.0 / 0.42) / vc.rgb_d[i]
        });
        rgb_from_xyz(mul(&CAM16RGB_TO_XYZ, cam_rgb))
    }
}

/// A colour described by hue (degrees), chroma and tone (L*, 0..=100).
#[derive(Clone, Copy, Debug)]
pub struct Hct {
    /// CAM16 hue in degrees.
    pub hue: f64,
    /// CAM16 chroma.
    pub chroma: f64,
    /// CIE L*.
    pub tone: f64,
}

impl Hct {
    /// Measure an sRGB colour.
    #[must_use]
    pub fn from_rgb(rgb: Rgb) -> Self {
        let cam = Cam16::from_rgb(rgb);
        Self { hue: cam.hue, chroma: cam.chroma, tone: lstar_from_rgb(rgb) }
    }

    /// The sRGB colour closest to `hue`/`chroma` at exactly `tone`,
    /// lowering chroma as far as needed to stay in gamut.
    #[must_use]
    pub fn solve(hue: f64, chroma: f64, tone: f64) -> Rgb {
        if chroma < 1.0 || tone.round() <= 0.0 || tone.round() >= 100.0 {
            return gray_from_lstar(tone);
        }
        let hue = sanitize_degrees(hue);
        if let Some(cam) = find_cam_by_j(hue, chroma, tone) {
            return cam.to_rgb();
        }
        let (mut low, mut high) = (0.0, chroma);
        let mut answer = None;
        while (high - low) >= CHROMA_SEARCH_TOLERANCE {
            let mid = low + (high - low) / 2.0;
            match find_cam_by_j(hue, mid, tone) {
                Some(cam) => {
                    answer = Some(cam);
                    low = mid;
                }
                None => high = mid,
            }
        }
        answer.map_or_else(|| gray_from_lstar(tone), Cam16::to_rgb)
    }
}

/// Bisect CAM16 lightness for a colour of the given hue/chroma whose clipped
/// sRGB rendering lands on `tone` without drifting in hue.
fn find_cam_by_j(hue: f64, chroma: f64, tone: f64) -> Option<Cam16> {
    let (mut low, mut high) = (0.0, 100.0);
    let mut best: Option<(f64, f64, Cam16)> = None;
    while (high - low) > J_SEARCH_TOLERANCE {
        let mid = low + (high - low) / 2.0;
        let clipped = Cam16::from_jch(mid, chroma, hue).to_rgb();
        let clipped_lstar = lstar_from_rgb(clipped);
        let tone_error = (tone - clipped_lstar).abs();
        if tone_error < TONE_MATCH_TOLERANCE {
            let actual = Cam16::from_rgb(clipped);
            let drift = actual.distance(&Cam16::from_jch(actual.j, actual.chroma, hue));
            let improves = best.is_none_or(|(_, best_drift, _)| drift <= best_drift);
            if drift <= MAX_HUE_DRIFT && improves {
                best = Some((tone_error, drift, actual));
            }
        }
        if best.is_some_and(|(err, drift, _)| err == 0.0 && drift == 0.0) {
            break;
        }
        if clipped_lstar < tone {
            low = mid;
        } else {
            high = mid;
        }
    }
    best.map(|(_, _, cam)| cam)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_baseline_purple() {
        let seed = [0x67, 0x50, 0xA4];
        let hct = Hct::from_rgb(seed);
        let back = Hct::solve(hct.hue, hct.chroma, hct.tone);
        for (a, b) in seed.iter().zip(back) {
            assert!(a.abs_diff(b) <= 2, "{seed:?} vs {back:?}");
        }
    }

    #[test]
    fn tone_is_lstar() {
        assert_eq!(Hct::solve(0.0, 0.0, 100.0), [255, 255, 255]);
        assert_eq!(Hct::solve(0.0, 0.0, 0.0), [0, 0, 0]);
        let mid = Hct::solve(270.0, 30.0, 50.0);
        assert!((lstar_from_rgb(mid) - 50.0).abs() < 0.5);
    }
}
