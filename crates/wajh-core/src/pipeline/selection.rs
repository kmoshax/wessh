//! Which detected faces get swapped, and in what order.

use crate::face::Face;

/// Order faces are considered in; with a single face, the first one wins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum FaceOrder {
    #[default]
    LargeToSmall,
    LeftToRight,
    RightToLeft,
    TopToBottom,
    /// Most confident detection first.
    BestScore,
}

impl FaceOrder {
    pub const ALL: [Self; 5] = [Self::LargeToSmall, Self::LeftToRight, Self::RightToLeft, Self::TopToBottom, Self::BestScore];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LargeToSmall => "Largest first",
            Self::LeftToRight => "Left to right",
            Self::RightToLeft => "Right to left",
            Self::TopToBottom => "Top to bottom",
            Self::BestScore => "Most confident",
        }
    }

    pub fn sort(self, faces: &mut [Face]) {
        let center_x = |f: &Face| f.bbox[0] + f.bbox[2];
        match self {
            Self::LargeToSmall => faces.sort_by(|a, b| b.size().total_cmp(&a.size())),
            Self::LeftToRight => faces.sort_by(|a, b| center_x(a).total_cmp(&center_x(b))),
            Self::RightToLeft => faces.sort_by(|a, b| center_x(b).total_cmp(&center_x(a))),
            Self::TopToBottom => faces.sort_by(|a, b| (a.bbox[1] + a.bbox[3]).total_cmp(&(b.bbox[1] + b.bbox[3]))),
            Self::BestScore => faces.sort_by(|a, b| b.score.total_cmp(&a.score)),
        }
    }
}

/// Order `faces` and keep those to swap.
///
/// With `similarities` (cosine similarity of each face to the reference identity), only faces
/// at least `min_similarity` alike are kept, and a single-face selection takes the closest match
/// rather than the first in `order`.
#[must_use]
pub fn select(faces: Vec<Face>, similarities: Option<&[f32]>, min_similarity: f32, order: FaceOrder, many: bool) -> Vec<Face> {
    let mut faces = match similarities {
        Some(similarities) => {
            let matches = faces.into_iter().zip(similarities.iter().copied()).filter(|(_, s)| *s >= min_similarity);
            if !many {
                return matches.max_by(|a, b| a.1.total_cmp(&b.1)).map(|(f, _)| f).into_iter().collect();
            }
            matches.map(|(f, _)| f).collect()
        }
        None => faces,
    };
    order.sort(&mut faces);
    if !many {
        faces.truncate(1);
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: f32, size: f32) -> Face {
        Face { bbox: [x, 0.0, x + size, size], score: 0.9, landmarks: [[x, 0.0]; 5] }
    }

    fn xs(faces: &[Face]) -> Vec<f32> {
        faces.iter().map(|f| f.bbox[0]).collect()
    }

    #[test]
    fn ordering_and_single_face() {
        let faces = vec![face(100.0, 10.0), face(0.0, 30.0), face(50.0, 20.0)];
        assert_eq!(xs(&select(faces.clone(), None, 0.0, FaceOrder::LeftToRight, true)), [0.0, 50.0, 100.0]);
        assert_eq!(xs(&select(faces.clone(), None, 0.0, FaceOrder::RightToLeft, false)), [100.0]);
        assert_eq!(xs(&select(faces, None, 0.0, FaceOrder::LargeToSmall, false)), [0.0]);
    }

    #[test]
    fn reference_filters_and_picks_the_closest() {
        let faces = vec![face(100.0, 10.0), face(0.0, 30.0), face(50.0, 20.0)];
        let sims = [0.7, 0.1, 0.5];
        assert_eq!(xs(&select(faces.clone(), Some(&sims), 0.4, FaceOrder::LeftToRight, true)), [50.0, 100.0]);
        assert_eq!(xs(&select(faces.clone(), Some(&sims), 0.4, FaceOrder::LargeToSmall, false)), [100.0]);
        assert_eq!(xs(&select(faces, Some(&sims), 0.9, FaceOrder::LargeToSmall, true)), [0.0; 0]);
    }
}
