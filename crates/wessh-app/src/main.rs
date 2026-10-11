//! wessh — real-time face swap.

// Double-clicking wessh.exe opens only the app window, no console; `wessh-cli.exe` (built with
// the `console` feature) is the same program for the terminal.
#![cfg_attr(all(windows, not(feature = "console")), windows_subsystem = "windows")]

mod cli;
mod commands;
mod install;
mod jobs;
mod live;
mod ui;
mod update;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    match cli.command {
        None => ui::run(cli.options),
        Some(command) => commands::run(command, &cli.options),
    }
}
