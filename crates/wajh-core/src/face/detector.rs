use super::Face;
use crate::assets;
use crate::image::{Frame, letterbox_nchw};
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
}

impl Detector {
    pub fn new(engine: &Engine) -> Result<Self> {
        Ok(Self { model: engine.load(&assets::DETECTOR)?, input_size: 640, score_threshold: 0.5 })
    }

    /// Detected faces, largest first.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Face>> {
        let size = self.input_size;
        let (tensor, to_frame) = letterbox_nchw(frame, size);
        let outputs = self.model.run(vec![Input::image("input", size, tensor)])?;
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
                faces.push(Face {
                    bbox: [(ax - d[0] * s), (ay - d[1] * s), (ax + d[2] * s), (ay + d[3] * s)].map(|v| v * to_frame),
                    score,
                    landmarks: std::array::from_fn(|j| [(ax + k[j * 2] * s) * to_frame, (ay + k[j * 2 + 1] * s) * to_frame]),
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
