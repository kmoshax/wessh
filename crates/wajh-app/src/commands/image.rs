use super::processor;
use crate::cli::Options;
use anyhow::Result;
use std::path::Path;
use wajh_core::Frame;

pub fn run(options: &Options, source: &Path, target: &Path, output: &Path) -> Result<()> {
    let mut processor = processor(options, source)?;
    let mut frame = Frame::open(target)?;
    let stats = processor.process(&mut frame, &options.settings(), false)?;
    frame.to_rgb().save(output)?;
    eprintln!("{} face(s) swapped in {:.1} ms → {}", stats.faces, stats.total_ms, output.display());
    Ok(())
}
