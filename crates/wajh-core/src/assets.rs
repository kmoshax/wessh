//! Model registry and first-run downloads.

use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

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

pub const DETECTOR: Asset = facefusion!("models-3.0.0", "scrfd_2.5g");
pub const RECOGNIZER: Asset = facefusion!("models-3.0.0", "arcface_w600k_r50");
pub const ENHANCER: Asset = facefusion!("models-3.0.0", "gpen_bfr_256");
/// `BiSeNet` face parser (19 `CelebAMask-HQ` classes, 512px).
pub const PARSER: Asset = facefusion!("models-3.0.0", "bisenet_resnet_34");

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

/// Face swap networks, fastest first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum SwapModel {
    /// InsightFace inswapper, 128px, fp16.
    #[default]
    Inswapper,
    /// FaceFusion HyperSwap 1a, 256px.
    Hyperswap1a,
    /// FaceFusion HyperSwap 1b, 256px.
    Hyperswap1b,
    /// FaceFusion HyperSwap 1c, 256px.
    Hyperswap1c,
}

impl SwapModel {
    pub const ALL: [Self; 4] = [Self::Inswapper, Self::Hyperswap1a, Self::Hyperswap1b, Self::Hyperswap1c];

    #[must_use]
    pub const fn asset(self) -> Asset {
        match self {
            Self::Inswapper => facefusion!("models-3.0.0", "inswapper_128_fp16"),
            Self::Hyperswap1a => facefusion!("models-3.3.0", "hyperswap_1a_256"),
            Self::Hyperswap1b => facefusion!("models-3.3.0", "hyperswap_1b_256"),
            Self::Hyperswap1c => facefusion!("models-3.3.0", "hyperswap_1c_256"),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Inswapper => "Inswapper 128",
            Self::Hyperswap1a => "HyperSwap 1a",
            Self::Hyperswap1b => "HyperSwap 1b",
            Self::Hyperswap1c => "HyperSwap 1c",
        }
    }

    /// Side length of the aligned crop the network works on.
    #[must_use]
    pub const fn resolution(self) -> usize {
        match self {
            Self::Inswapper => 128,
            Self::Hyperswap1a | Self::Hyperswap1b | Self::Hyperswap1c => 256,
        }
    }
}

/// Root of wajh's per-user data (`$WAJH_HOME`, or the platform data dir).
#[must_use]
pub fn data_dir() -> PathBuf {
    std::env::var_os("WAJH_HOME").map_or_else(|| dirs::data_local_dir().unwrap_or_else(|| PathBuf::from(".")).join("wajh"), PathBuf::from)
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
    let status = Command::new("curl")
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
