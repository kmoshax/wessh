use super::Face;
use crate::assets;
use crate::geometry::Rect;
use crate::image::{Frame, letterbox_region_nchw_into};
use crate::runtime::{Engine, StaticModel, StaticShapes};
use crate::{Error, Result};
use std::collections::HashMap;

/// Anchor strides of SCRFD's three feature levels.
const STRIDES: [usize; 3] = [8, 16, 32];
const ANCHORS_PER_CELL: usize = 2;
const NMS_IOU: f32 = 0.4;
/// Values per anchor in the score, box and keypoint outputs.
const SCORE_LEN: usize = 1;
const BOX_LEN: usize = 4;
const KEYPOINT_LEN: usize = 10;
/// Name of SCRFD's symbolic input height/width dimension.
const SIZE_DIMENSION: &str = "?";

/// SCRFD 10G face detector with 5-point landmarks.
///
/// One bound session is kept per input size: a fixed shape lets ONNX Runtime fold SCRFD's shape
/// arithmetic, keep every node on the GPU and replay the network as a CUDA graph, which matters
/// because the detector is launch-bound rather than compute-bound.
pub struct Detector {
    engine: Engine,
    models: HashMap<usize, StaticModel>,
    pub input_size: usize,
    pub score_threshold: f32,
}

impl Detector {
    /// Input sizes must be a multiple of the coarsest anchor stride.
    pub const SIZE_STEP: usize = STRIDES[STRIDES.len() - 1];
    pub const DEFAULT_SIZE: usize = 640;

    pub fn new(engine: &Engine) -> Result<Self> {
        let mut detector = Self { engine: engine.clone(), models: HashMap::new(), input_size: Self::DEFAULT_SIZE, score_threshold: 0.5 };
        detector.prepare(Self::DEFAULT_SIZE)?;
        Ok(detector)
    }

    /// Load (and warm up) the session for `size` ahead of time, so the first frame using it
    /// isn't slow.
    pub fn prepare(&mut self, size: usize) -> Result<()> {
        self.model(size).map(drop)
    }

    fn model(&mut self, size: usize) -> Result<&mut StaticModel> {
        let size = snap_size(size);
        if !self.models.contains_key(&size) {
            let outputs = output_shapes(size);
            let dimensions = [(SIZE_DIMENSION, size as i64)];
            let model = self.engine.load_static_with(
                &assets::DETECTOR,
                &StaticShapes { dimensions: &dimensions, outputs: &outputs, ..StaticShapes::default() },
            )?;
            self.models.insert(size, model);
        }
        Ok(self.models.get_mut(&size).expect("inserted above"))
    }

    /// Detected faces, largest first.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Face>> {
        self.detect_region(frame, Rect::full(frame.width, frame.height), self.input_size)
    }

    /// Faces inside `region` of `frame` (in frame coordinates), largest first, with the region
    /// letterboxed into a `size`×`size` input (rounded to a multiple of [`Self::SIZE_STEP`]).
    pub fn detect_region(&mut self, frame: &Frame, region: Rect, size: usize) -> Result<Vec<Face>> {
        let size = snap_size(size);
        let threshold = self.score_threshold;
        let model = self.model(size)?;
        let input = model.input_mut(0);
        let to_frame = letterbox_region_nchw_into(frame, region, size, input);
        // InsightFace trained SCRFD 10G on RGB (`blobFromImage(.., swapRB=True)`); the letterbox is BGR.
        let (blue, green_red) = input.split_at_mut(size * size);
        blue.swap_with_slice(&mut green_red[size * size..]);
        model.run()?;
        let outputs: Vec<&[f32]> = (0..STRIDES.len() * 3).map(|i| model.output(i)).collect();
        let transform = TensorToFrame { scale: to_frame, origin: [region.x as f32, region.y as f32] };
        let mut faces = non_max_suppression(decode(&outputs, size, threshold, transform)?);
        faces.sort_by(|a, b| b.size().total_cmp(&a.size()));
        Ok(faces)
    }
}

/// Maps detector-input coordinates to frame coordinates: `origin + p * scale`.
#[derive(Clone, Copy, Debug)]
struct TensorToFrame {
    scale: f32,
    origin: [f32; 2],
}

/// Round `size` to the nearest usable input size (a positive multiple of the coarsest stride).
fn snap_size(size: usize) -> usize {
    (size.div_ceil(Detector::SIZE_STEP) * Detector::SIZE_STEP).max(Detector::SIZE_STEP)
}

/// Output shapes for a `size`×`size` input: scores, boxes, keypoints, each per stride.
fn output_shapes(size: usize) -> Vec<Vec<usize>> {
    let anchors = |stride: usize| (size / stride).pow(2) * ANCHORS_PER_CELL;
    [SCORE_LEN, BOX_LEN, KEYPOINT_LEN].into_iter().flat_map(|len| STRIDES.map(|stride| vec![anchors(stride), len])).collect()
}

/// Decode SCRFD's anchor-relative outputs into faces above `threshold`.
fn decode(outputs: &[&[f32]], size: usize, threshold: f32, transform: TensorToFrame) -> Result<Vec<Face>> {
    if outputs.len() != STRIDES.len() * 3 {
        return Err(Error::UnexpectedOutput { model: assets::DETECTOR.name, detail: format!("{} outputs", outputs.len()) });
    }
    let to_x = |v: f32| v.mul_add(transform.scale, transform.origin[0]);
    let to_y = |v: f32| v.mul_add(transform.scale, transform.origin[1]);
    let mut faces = Vec::new();
    for (level, stride) in STRIDES.into_iter().enumerate() {
        let (scores, boxes, points) = (outputs[level], outputs[level + 3], outputs[level + 6]);
        let columns = size / stride;
        let s = stride as f32;
        for (i, &score) in scores.iter().enumerate().filter(|&(_, &score)| score >= threshold) {
            let cell = i / ANCHORS_PER_CELL;
            let (ax, ay) = (((cell % columns) * stride) as f32, ((cell / columns) * stride) as f32);
            let (Some(d), Some(k)) = (boxes.get(i * BOX_LEN..(i + 1) * BOX_LEN), points.get(i * KEYPOINT_LEN..(i + 1) * KEYPOINT_LEN))
            else {
                return Err(Error::UnexpectedOutput { model: assets::DETECTOR.name, detail: "output lengths disagree".into() });
            };
            faces.push(Face {
                bbox: [to_x(ax - d[0] * s), to_y(ay - d[1] * s), to_x(ax + d[2] * s), to_y(ay + d[3] * s)],
                score,
                landmarks: std::array::from_fn(|j| [to_x(ax + k[j * 2] * s), to_y(ay + k[j * 2 + 1] * s)]),
            });
        }
    }
    Ok(faces)
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
    use super::*;

    #[test]
    fn iou_of_identical_and_disjoint_boxes() {
        let a = [0.0, 0.0, 10.0, 10.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        assert!(iou(&a, &[20.0, 20.0, 30.0, 30.0]).abs() < 1e-6);
    }

    #[test]
    fn sizes_snap_to_the_stride() {
        assert_eq!(snap_size(640), 640);
        assert_eq!(snap_size(600), 608);
        assert_eq!(snap_size(0), 32);
    }

    #[test]
    fn output_shapes_follow_the_anchor_grid() {
        let shapes = output_shapes(64);
        assert_eq!(shapes.len(), 9);
        assert_eq!(shapes[0], [(64 / 8) * (64 / 8) * 2, 1]);
        assert_eq!(shapes[5], [2 * 2 * 2, 4]);
        assert_eq!(shapes[8], [2 * 2 * 2, 10]);
    }

    #[test]
    fn decode_maps_an_anchor_into_the_region() {
        let size = 32;
        let shapes = output_shapes(size);
        let mut data: Vec<Vec<f32>> = shapes.iter().map(|s| vec![0.0; s.iter().product()]).collect();
        // Anchor 0 of stride 8 (at the origin): a 2-stride box and all keypoints at (1, 1) strides.
        data[0][0] = 0.9;
        data[3][..4].copy_from_slice(&[0.0, 0.0, 2.0, 2.0]);
        data[6][..10].fill(1.0);
        let outputs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
        let faces = decode(&outputs, size, 0.5, TensorToFrame { scale: 2.0, origin: [100.0, 50.0] }).expect("decodes");
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].bbox, [100.0, 50.0, 132.0, 82.0]);
        assert_eq!(faces[0].landmarks[0], [116.0, 66.0]);
    }
}
