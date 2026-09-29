use super::Face;
use crate::assets;
use crate::geometry::Rect;
use crate::image::{Frame, letterbox_region_nchw_into};
use crate::runtime::{Engine, Input, Model};
use crate::{Error, Result};

const STRIDES: [usize; 3] = [8, 16, 32];
const ANCHORS_PER_CELL: usize = 2;
const NMS_IOU: f32 = 0.4;

/// SCRFD face detector with 5-point landmarks.
pub struct Detector {
    model: Model,
    pub input_size: usize,
    pub score_threshold: f32,
    /// Reused input tensor.
    tensor: Vec<f32>,
}

impl Detector {
    /// Input sizes must be a multiple of the coarsest anchor stride.
    pub const SIZE_STEP: usize = 32;

    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load(&assets::DETECTOR)?, input_size: 640, score_threshold: 0.5, tensor: Vec::new() })
    }

    /// Detected faces, largest first.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Face>> {
        self.detect_region(frame, Rect::full(frame.width, frame.height), self.input_size)
    }

    /// Faces inside `region` of `frame` (in frame coordinates), largest first, with the region
    /// letterboxed into a `size`×`size` input (`size` a multiple of [`Self::SIZE_STEP`]).
    pub fn detect_region(&mut self, frame: &Frame, region: Rect, size: usize) -> Result<Vec<Face>> {
        self.tensor.resize(size * size * 3, 0.0);
        let to_frame = letterbox_region_nchw_into(frame, region, size, &mut self.tensor);
        let origin = [region.x as f32, region.y as f32];
        let outputs = self.model.run(&[Input::image("input", size, &self.tensor)])?;
        if outputs.len() != STRIDES.len() * 3 {
            return Err(Error::UnexpectedOutput { model: assets::DETECTOR.name, detail: format!("{} outputs", outputs.len()) });
        }

        let mut faces = Vec::new();
        for (level, stride) in STRIDES.into_iter().enumerate() {
            let (scores, boxes, points) = (&outputs[level], &outputs[level + 3], &outputs[level + 6]);
            let columns = size / stride;
            let s = stride as f32;
            for (i, &score) in scores.iter().enumerate().filter(|&(_, &score)| score >= self.score_threshold) {
                let cell = i / ANCHORS_PER_CELL;
                let (ax, ay) = (((cell % columns) * stride) as f32, ((cell / columns) * stride) as f32);
                let d = &boxes[i * 4..i * 4 + 4];
                let k = &points[i * 10..i * 10 + 10];
                let to_x = |v: f32| v.mul_add(to_frame, origin[0]);
                let to_y = |v: f32| v.mul_add(to_frame, origin[1]);
                faces.push(Face {
                    bbox: [to_x(ax - d[0] * s), to_y(ay - d[1] * s), to_x(ax + d[2] * s), to_y(ay + d[3] * s)],
                    score,
                    landmarks: std::array::from_fn(|j| [to_x(ax + k[j * 2] * s), to_y(ay + k[j * 2 + 1] * s)]),
                });
            }
        }
        let mut faces = non_max_suppression(faces);
        faces.sort_by(|a, b| b.size().total_cmp(&a.size()));
        Ok(faces)
    }
}

fn non_max_suppression(mut faces: Vec<Face>) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::with_capacity(faces.len());
    for face in faces {
        if kept.iter().all(|k| iou(&k.bbox, &face.bbox) < NMS_IOU) {
            kept.push(face);
        }
    }
    kept
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let intersection = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    intersection / (area(a) + area(b) - intersection).max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::iou;

    #[test]
    fn iou_of_identical_and_disjoint_boxes() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        assert!(iou(&a, &[20.0, 20.0, 30.0, 30.0]).abs() < 1e-6);
    }
}
