//! Headless subcommands.

mod bench;
mod cam;
mod image;
mod models;
mod video;

use crate::cli::{Command, Options};
use anyhow::Result;
use std::path::Path;
use wajh_core::{Engine, Frame, Processor};

pub fn run(command: Command, options: &Options) -> Result<()> {
    match command {
        Command::Image { source, target, output } => image::run(options, &source, &target, &output),
        Command::Video { source, target, output, encoder } => video::run(options, &source, &target, &output, encoder),
        Command::Cam { source, camera, out } => cam::run(options, &source, &camera, out.as_deref()),
        Command::Bench { source, target, iterations, profile } => bench::run(options, &source, &target, iterations, profile),
        Command::Models => models::run(options),
    }
}

/// Engine + processor with every model loaded and the source identity set.
fn processor(options: &Options, source: &Path) -> Result<Processor> {
    let engine = Engine::init(options.ort.as_deref(), options.device)?;
    eprintln!("backend: {}", engine.backend());
    let mut processor = Processor::new(engine)?;
    processor.prepare(&options.settings())?;
    processor.set_source(&Frame::open(source)?)?;
    Ok(processor)
}
