//! wajh — real-time face swap.

mod cli;
mod commands;
mod live;
mod ui;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    match cli.command {
        None => ui::run(cli.options),
        Some(command) => commands::run(command, &cli.options),
    }
}
