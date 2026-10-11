//! Model registry and first-run downloads.

use crate::process::background_command;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// A downloadable ONNX model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Asset {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
}

macro_rules! facefusion {
    ($repo:literal, $name:literal) => {
        Asset {
            name: $name,
            file: concat!($name, ".onnx"),
            url: concat!("https://huggingface.co/facefusion/", $repo, "/resolve/main/", $name, ".onnx"),
        }
    };
}

/// A model from Deep-Live-Cam's repository, stored under our own file name.
macro_rules! deep_live_cam {
    ($path:literal, $name:literal) => {
        Asset {
            name: $name,
            file: concat!($name, ".onnx"),
            url: concat!("https://huggingface.co/hacksider/deep-live-cam/resolve/main/", $path),
        }
    };
}

/// InsightFace SCRFD 10G (`buffalo_l`), the detector Deep-Live-Cam uses. Its landmarks are far
/// more precise than SCRFD 2.5G's, and the landmarks decide how well the swapped face sits.
pub const DETECTOR: Asset = deep_live_cam!("buffalo_l/buffalo_l/det_10g.onnx", "det_10g");
pub const RECOGNIZER: Asset = facefusion!("models-3.0.0", "arcface_w600k_r50");
/// GFPGAN 1.4 (512 px in, 1024 px out), the restorer Deep-Live-Cam uses.
pub const ENHANCER: Asset = deep_live_cam!("gfpgan-1024.onnx", "gfpgan_1024");
/// `BiSeNet` face parser (19 `CelebAMask-HQ` classes, 512px).
pub const PARSER: Asset = facefusion!("models-3.0.0", "bisenet_resnet_34");
/// The lighter `BiSeNet` backbone, same classes and resolution.
pub const PARSER_LITE: Asset = facefusion!("models-3.1.0", "bisenet_resnet_18");

/// `XSeg` occluders: segment the visible face so hands, hair and glasses in front of it survive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum OccluderModel {
    #[default]
    Xseg1,
    Xseg2,
    Xseg3,
}

impl OccluderModel {
    pub const ALL: [Self; 3] = [Self::Xseg1, Self::Xseg2, Self::Xseg3];

    #[must_use]
    pub const fn asset(self) -> Asset {
        match self {
            Self::Xseg1 => facefusion!("models-3.1.0", "xseg_1"),
            Self::Xseg2 => facefusion!("models-3.1.0", "xseg_2"),
            Self::Xseg3 => facefusion!("models-3.2.0", "xseg_3"),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Xseg1 => "XSeg 1",
            Self::Xseg2 => "XSeg 2",
            Self::Xseg3 => "XSeg 3",
        }
    }
}

/// Face swap networks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum SwapModel {
    /// InsightFace inswapper, 128px, fp16. The default: across a 4 × 14 source/target set it
    /// carried the source identity far better than HyperSwap (`ArcFace` similarity 0.84 vs 0.74)
    /// and left less of the target behind, at the same score as Deep-Live-Cam.
    #[default]
    Inswapper,
    /// FaceFusion HyperSwap 1a, 256px. Sharper and, with TensorRT, faster (≈10 ms vs ≈22 ms), but
    /// a weaker identity transfer.
    Hyperswap1a,
    /// FaceFusion HyperSwap 1b, 256px.
    Hyperswap1b,
    /// FaceFusion HyperSwap 1c, 256px.
    Hyperswap1c,
    /// AlphaFace (2026), 256px; better on turned heads.
    Alphaface,
}

impl SwapModel {
    pub const ALL: [Self; 5] = [Self::Inswapper, Self::Hyperswap1a, Self::Hyperswap1b, Self::Hyperswap1c, Self::Alphaface];

    #[must_use]
    pub const fn asset(self) -> Asset {
        match self {
            Self::Inswapper => facefusion!("models-3.0.0", "inswapper_128_fp16"),
            Self::Hyperswap1a => facefusion!("models-3.3.0", "hyperswap_1a_256"),
            Self::Hyperswap1b => facefusion!("models-3.3.0", "hyperswap_1b_256"),
            Self::Hyperswap1c => facefusion!("models-3.3.0", "hyperswap_1c_256"),
            Self::Alphaface => facefusion!("models-3.9.0", "alphaface_256"),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Inswapper => "Inswapper 128",
            Self::Hyperswap1a => "HyperSwap 1a",
            Self::Hyperswap1b => "HyperSwap 1b",
            Self::Hyperswap1c => "HyperSwap 1c",
            Self::Alphaface => "AlphaFace",
        }
    }

    /// Whether TensorRT's fp16 build reproduces the network's output. inswapper and `AlphaFace`
    /// come out washed-out and noisy (checked against the CUDA provider on an RTX 4050), and
    /// inswapper's fp32 build is slower than CUDA (34 ms vs 22 ms), so both stay on CUDA.
    #[must_use]
    pub const fn tensorrt_safe(self) -> bool {
        matches!(self, Self::Hyperswap1a | Self::Hyperswap1b | Self::Hyperswap1c)
    }

    /// Side length of the aligned crop the network works on.
    #[must_use]
    pub const fn resolution(self) -> usize {
        match self {
            Self::Inswapper => 128,
            Self::Hyperswap1a | Self::Hyperswap1b | Self::Hyperswap1c | Self::Alphaface => 256,
        }
    }
}

/// Models needed by the enabled features. Disabled enhancement and masks are downloaded only
/// when requested; the parser is shared by region and mouth masks.
#[must_use]
pub fn required(settings: &crate::Settings) -> Vec<Asset> {
    let mut models = vec![DETECTOR, RECOGNIZER, settings.swap_model.asset()];
    if settings.enhance {
        models.push(ENHANCER);
    }
    if settings.occlusion {
        models.push(settings.occluder_model.asset());
    }
    if settings.region_mask || settings.mouth_mask {
        models.push(crate::face::masker::PARSER_ASSET);
    }
    models
}

/// Root of wessh's per-user data (`$WESSH_HOME`, or the platform data dir).
#[must_use]
pub fn data_dir() -> PathBuf {
    std::env::var_os("WESSH_HOME").map_or_else(|| dirs::data_local_dir().unwrap_or_else(|| PathBuf::from(".")).join("wessh"), PathBuf::from)
}

#[must_use]
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

impl Asset {
    #[must_use]
    pub fn path(&self) -> PathBuf {
        models_dir().join(self.file)
    }

    #[must_use]
    pub fn is_downloaded(&self) -> bool {
        self.path().exists()
    }

    /// Local path of the model, downloading it first if needed (resumable).
    pub fn ensure(&self) -> Result<PathBuf> {
        let path = self.path();
        if !path.exists() {
            download(self.url, &path)?;
        }
        Ok(path)
    }
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let fail = |reason: String| Error::Download { url: url.to_owned(), reason };
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = dest.with_extension("part");
    let status = background_command("curl")
        .args(["-fL", "--retry", "5", "-C", "-", "--progress-bar", "-o"])
        .arg(&partial)
        .arg(url)
        .status()
        .map_err(|e| fail(format!("could not run curl: {e}")))?;
    if !status.success() {
        return Err(fail(format!("curl exited with {status}")));
    }
    std::fs::rename(&partial, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Settings;

    #[test]
    fn default_downloads_skip_disabled_masks() {
        assert_eq!(required(&Settings::default()), [DETECTOR, RECOGNIZER, SwapModel::Inswapper.asset(), ENHANCER]);
        let settings = Settings { enhance: false, ..Settings::default() };
        assert_eq!(required(&settings), [DETECTOR, RECOGNIZER, SwapModel::Inswapper.asset()]);
    }

    #[test]
    fn optional_models_follow_settings_without_duplicate_parser() {
        let settings = Settings {
            enhance: false,
            swap_model: SwapModel::Hyperswap1c,
            occlusion: true,
            occluder_model: OccluderModel::Xseg3,
            region_mask: true,
            mouth_mask: true,
            ..Settings::default()
        };
        assert_eq!(
            required(&settings),
            [DETECTOR, RECOGNIZER, SwapModel::Hyperswap1c.asset(), OccluderModel::Xseg3.asset(), crate::face::masker::PARSER_ASSET]
        );
        let mouth_only = Settings { mouth_mask: true, enhance: false, ..Settings::default() };
        assert!(required(&mouth_only).contains(&crate::face::masker::PARSER_ASSET));
    }
}
