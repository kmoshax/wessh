//! Headless subcommands.

mod bench;
mod cam;
pub mod camera;
mod doctor;
mod image;
mod models;
mod setup;
mod video;

use crate::cli::{Command, Options};
use anyhow::Result;
use std::path::PathBuf;
use wessh_core::{Engine, Frame, Processor};

pub fn run(command: Command, options: &Options) -> Result<()> {
    match command {
        Command::Image { source, target, output } => image::run(options, &source, &target, &output),
        Command::Video { source, target, output, encoder } => video::run(options, &source, &target, &output, encoder),
        Command::Cam { source, camera, out } => cam::run(options, &source, &camera, out.as_deref()),
        Command::Bench { source, target, iterations, profile, motion } => {
            bench::run(options, &source, &target, iterations, profile, motion)
        }
        Command::Models => models::run(options),
        Command::Setup(args) => setup::run(&args, options),
        Command::Doctor => doctor::run(options),
        Command::Camera { action } => camera::run(action),
    }
}

/// Engine + processor with every model loaded, the source identity and any reference face set.
fn processor(options: &Options, sources: &[PathBuf]) -> Result<Processor> {
    let engine = Engine::init(options.ort.as_deref(), options.device)?;
    eprintln!("backend: {}", engine.backend());
    if let Some(reason) = engine.gpu_failure() {
        eprintln!("warning: running on the CPU because {reason}; `wessh doctor` explains how to fix it");
    }
    let mut processor = Processor::new(engine, &options.settings())?;
    let sources = sources.iter().map(|path| Frame::open(path)).collect::<wessh_core::Result<Vec<_>>>()?;
    processor.set_sources(&sources)?;
    if let Some(reference) = &options.reference {
        processor.set_reference(&Frame::open(reference)?, options.face_order, options.reference_face)?;
    }
    Ok(processor)
}
