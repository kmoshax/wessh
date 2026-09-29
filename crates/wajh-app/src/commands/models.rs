use crate::cli::Options;
use anyhow::Result;
use wajh_core::assets;

pub fn run(options: &Options) -> Result<()> {
    let mut needed = vec![assets::DETECTOR, assets::RECOGNIZER, options.model.asset()];
    if options.enhance {
        needed.push(assets::ENHANCER);
    }
    for asset in needed {
        println!("{}", asset.ensure()?.display());
    }
    Ok(())
}
