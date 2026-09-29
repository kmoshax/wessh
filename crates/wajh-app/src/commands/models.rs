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
    let mut needed = vec![assets::DETECTOR, assets::RECOGNIZER, options.model.asset()];
    if options.enhance {
        needed.push(assets::ENHANCER);
    }
    needed
}
