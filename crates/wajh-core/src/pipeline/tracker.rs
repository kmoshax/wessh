//! Cheap per-frame face localisation for live video.
//!
//! A full-frame detection finds the faces; on the following frames each face is re-detected
//! only inside a small square around where it was, at a small detector input. That keeps the
//! landmarks exact for the current frame (no extrapolation, so no added lag) while the detector
//! costs a fraction of a full pass. A full pass runs again periodically (to pick up new faces)
//! and immediately whenever a tracked face isn't found in its region — large motion, occlusion,
//! a scene cut.

use crate::Result;
use crate::face::{Detector, Face};
use crate::geometry::{Point, Rect};
use crate::image::Frame;

/// Where the tracker gets detections from; implemented by [`Detector`], faked in tests.
pub trait FaceSource {
    /// Every face in the frame, largest first.
    fn detect_full(&mut self, frame: &Frame) -> Result<Vec<Face>>;
    /// Faces inside `region`, largest first, in frame coordinates.
    fn detect_in(&mut self, frame: &Frame, region: Rect) -> Result<Vec<Face>>;
}

/// [`Detector`] with the region input size used for tracking.
pub struct TrackingDetector<'a> {
    pub detector: &'a mut Detector,
    pub region_size: usize,
}

impl FaceSource for TrackingDetector<'_> {
    fn detect_full(&mut self, frame: &Frame) -> Result<Vec<Face>> {
        self.detector.detect(frame)
    }

    fn detect_in(&mut self, frame: &Frame, region: Rect) -> Result<Vec<Face>> {
        self.detector.detect_region(frame, region, self.region_size)
    }
}

/// Tuning of the [`FaceTracker`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Side of the search square, in face sizes. A face can move `(scale - 1) / 2` face sizes
    /// between frames and still be found.
    pub region_scale: f32,
    /// Detector input size for a search square. The face then spans about
    /// `region_size / region_scale` pixels, plenty for SCRFD's landmarks.
    pub region_size: usize,
    /// Frames between full detections, which discover faces that entered the picture.
    pub full_detection_interval: u32,
    /// With more faces than this, one full pass is cheaper than one pass per face.
    pub max_regions: usize,
    /// A region detection only counts if it lies within this many face sizes of the track.
    pub max_jump: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self { region_scale: 2.2, region_size: 192, full_detection_interval: 30, max_regions: 2, max_jump: 0.6 }
    }
}

/// Which detection pass produced the faces of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Full,
    Regions,
}

/// Follows faces from frame to frame with region-limited detection.
#[derive(Debug, Default)]
pub struct FaceTracker {
    pub config: TrackerConfig,
    faces: Vec<Face>,
    frames_since_full: u32,
}

impl FaceTracker {
    /// Forget every face; the next frame gets a full detection.
    pub fn reset(&mut self) {
        self.faces.clear();
    }

    /// Faces in `frame`, largest first: the tracked faces re-found in their regions, or every
    /// face from a full pass. Report which of them to keep following with [`FaceTracker::track`].
    pub fn detect(&mut self, source: &mut impl FaceSource, frame: &Frame) -> Result<(Vec<Face>, Pass)> {
        let due = self.faces.is_empty()
            || self.faces.len() > self.config.max_regions
            || self.frames_since_full >= self.config.full_detection_interval;
        if !due && let Some(faces) = self.detect_regions(source, frame)? {
            self.frames_since_full += 1;
            return Ok((faces, Pass::Regions));
        }
        self.frames_since_full = 0;
        Ok((source.detect_full(frame)?, Pass::Full))
    }

    /// Follow `faces` (typically the ones selected for swapping) from the next frame on.
    pub fn track(&mut self, faces: &[Face]) {
        self.faces.clear();
        self.faces.extend_from_slice(faces);
    }

    /// Re-find every tracked face in its search square; `None` if any of them is lost.
    fn detect_regions(&self, source: &mut impl FaceSource, frame: &Frame) -> Result<Option<Vec<Face>>> {
        let mut found = Vec::with_capacity(self.faces.len());
        for previous in &self.faces {
            let size = previous.size();
            let Some(region) = Rect::square_around(center(previous), size * self.config.region_scale, frame.width, frame.height) else {
                return Ok(None);
            };
            let candidates = source.detect_in(frame, region)?;
            match closest(previous, &candidates, self.config.max_jump) {
                Some(face) => found.push(face.clone()),
                None => return Ok(None),
            }
        }
        found.sort_by(|a, b| b.size().total_cmp(&a.size()));
        Ok(Some(found))
    }
}

fn center(face: &Face) -> Point {
    [f32::midpoint(face.bbox[0], face.bbox[2]), f32::midpoint(face.bbox[1], face.bbox[3])]
}

/// The candidate nearest to `previous`, if it is within `max_jump` face sizes.
fn closest<'a>(previous: &Face, candidates: &'a [Face], max_jump: f32) -> Option<&'a Face> {
    let [px, py] = center(previous);
    let limit = previous.size() * max_jump;
    candidates
        .iter()
        .map(|c| {
            let [cx, cy] = center(c);
            (c, (cx - px).hypot(cy - py))
        })
        .filter(|&(_, distance)| distance <= limit)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(c, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_at(x: f32, y: f32, side: f32) -> Face {
        let h = side / 2.0;
        Face { bbox: [x - h, y - h, x + h, y + h], score: 0.9, landmarks: [[x, y]; 5] }
    }

    /// Faces at fixed positions; region detection returns those whose centre is inside.
    struct Scene {
        faces: Vec<Face>,
        full_passes: usize,
        region_passes: usize,
    }

    impl FaceSource for Scene {
        fn detect_full(&mut self, _: &Frame) -> Result<Vec<Face>> {
            self.full_passes += 1;
            Ok(self.faces.clone())
        }

        fn detect_in(&mut self, _: &Frame, r: Rect) -> Result<Vec<Face>> {
            self.region_passes += 1;
            let inside = |f: &&Face| {
                let [x, y] = center(f);
                (r.x as f32..(r.x + r.width) as f32).contains(&x) && (r.y as f32..(r.y + r.height) as f32).contains(&y)
            };
            Ok(self.faces.iter().filter(inside).cloned().collect())
        }
    }

    /// Detect, then keep tracking everything found.
    fn step(tracker: &mut FaceTracker, source: &mut Scene, frame: &Frame) -> (Vec<Face>, Pass) {
        let (faces, pass) = tracker.detect(source, frame).expect("detect");
        tracker.track(&faces);
        (faces, pass)
    }

    fn scene(faces: Vec<Face>) -> Scene {
        Scene { faces, full_passes: 0, region_passes: 0 }
    }

    #[test]
    fn follows_a_moving_face_with_region_passes() {
        let frame = Frame::new(640, 480);
        let mut tracker = FaceTracker::default();
        let mut source = scene(vec![face_at(300.0, 200.0, 100.0)]);
        assert_eq!(step(&mut tracker, &mut source, &frame).1, Pass::Full);
        source.faces = vec![face_at(330.0, 210.0, 100.0)];
        let (faces, pass) = step(&mut tracker, &mut source, &frame);
        assert_eq!((pass, center(&faces[0])), (Pass::Regions, [330.0, 210.0]));
        assert_eq!((source.full_passes, source.region_passes), (1, 1));
    }

    #[test]
    fn large_motion_triggers_a_full_pass() {
        let frame = Frame::new(640, 480);
        let mut tracker = FaceTracker::default();
        let mut source = scene(vec![face_at(150.0, 200.0, 80.0)]);
        step(&mut tracker, &mut source, &frame);
        source.faces = vec![face_at(500.0, 200.0, 80.0)];
        let (faces, pass) = step(&mut tracker, &mut source, &frame);
        assert_eq!((pass, center(&faces[0])), (Pass::Full, [500.0, 200.0]));
    }

    #[test]
    fn full_pass_runs_periodically_and_with_many_faces() {
        let frame = Frame::new(640, 480);
        let mut tracker =
            FaceTracker { config: TrackerConfig { full_detection_interval: 2, ..TrackerConfig::default() }, ..FaceTracker::default() };
        let mut source = scene(vec![face_at(300.0, 200.0, 100.0)]);
        let passes: Vec<Pass> = (0..4).map(|_| step(&mut tracker, &mut source, &frame).1).collect();
        assert_eq!(passes, [Pass::Full, Pass::Regions, Pass::Regions, Pass::Full]);

        let mut crowd = scene((0..3).map(|i| face_at(100.0 + 200.0 * i as f32, 200.0, 60.0)).collect());
        tracker.reset();
        step(&mut tracker, &mut crowd, &frame);
        assert_eq!(step(&mut tracker, &mut crowd, &frame).1, Pass::Full);
    }

    #[test]
    fn closest_ignores_far_candidates() {
        let previous = face_at(100.0, 100.0, 50.0);
        let candidates = [face_at(200.0, 100.0, 50.0), face_at(110.0, 100.0, 50.0)];
        assert_eq!(closest(&previous, &candidates, 0.6).map(center), Some([110.0, 100.0]));
        assert!(closest(&previous, &candidates[..1], 0.6).is_none());
    }
}
