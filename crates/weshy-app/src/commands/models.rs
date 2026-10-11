use crate::cli::Options;
use anyhow::Result;
use weshy_core::assets;

/// Download the curated model set ([`assets::required`]), the same list setup and the app's
/// onboarding install, and print where each model is.
pub fn run(options: &Options) -> Result<()> {
    for asset in assets::required(&options.settings()) {
        println!("{}", asset.ensure()?.display());
    }
    Ok(())
}
