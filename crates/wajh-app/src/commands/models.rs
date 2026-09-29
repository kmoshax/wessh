use crate::cli::Options;
use anyhow::Result;
use wajh_core::assets::{self, Asset};

pub fn run(options: &Options) -> Result<()> {
    for asset in needed(options) {
        println!("{}", asset.ensure()?.display());
    }
    Ok(())
}

/// Models the chosen options need.
pub fn needed(options: &Options) -> Vec<Asset> {
    let settings = options.settings();
    let mut needed = vec![assets::DETECTOR, assets::RECOGNIZER, settings.swap_model.asset()];
    if settings.enhance {
        needed.push(assets::ENHANCER);
    }
    if settings.occlusion {
        needed.push(settings.occluder_model.asset());
    }
    if settings.region_mask || settings.mouth_mask {
        needed.push(assets::PARSER);
    }
    needed
}
