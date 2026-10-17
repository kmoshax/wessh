# Contributing to wessh

Thanks for helping. wessh aims to be the fastest and easiest-to-install face swapper, so
changes are judged on two things above all: latency and install friction.

## Getting started

You need Rust 1.99 (pinned in `rust-toolchain.toml`, which rustup picks up), `curl`, and `ffmpeg`
on `PATH`.

```sh
git clone https://github.com/kmoshax/wessh && cd wessh
cargo build
# Runtime, CUDA libraries and models into a throwaway data dir, leaving your own install alone:
WESSH_HOME=/tmp/wessh-dev cargo run -- setup
WESSH_HOME=/tmp/wessh-dev cargo run -- doctor
WESSH_HOME=/tmp/wessh-dev cargo run --release            # the app
```

`cargo run -- setup --dry-run` shows what would be downloaded without touching anything.

## Before opening a pull request

CI runs exactly these on Linux, Windows and macOS:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

- **Formatting**: `rustfmt.toml` sets a 140-column width; run `cargo fmt --all`.
- **Lints**: the workspace enables `clippy::pedantic`. Fix warnings rather than silencing
  them; if an `#[allow]` is really right, scope it to the item and give a `reason`.
- **Tests**: pure logic (geometry, parsing, selection rules, archive filtering…) gets unit
  tests next to the code. Tests must not need a GPU, models or network access.
- **Commits**: small and focused, with an imperative subject line ("Add mouth mask").

## Code conventions

- Libraries (`wessh-core`, `wessh-media`) return typed errors (`thiserror`); the binary
  (`wessh-app`) uses `anyhow` and adds context.
- No magic numbers: name constants and say where they come from (a model's input size, a
  driver version, a protocol limit).
- No `unsafe` (the workspace forbids it).
- Keep the hot path allocation-free and avoid whole-frame copies; measure before and after
  (see below).
- Dependencies must earn their place: the release binary is a few megabytes and should stay that way.
  Prefer the standard library, then small well-maintained crates.

## Architecture

| Crate         | Role                                                                          |
| ------------- | ----------------------------------------------------------------------------- |
| `wessh-core`  | the face pipeline on ONNX Runtime; no I/O besides model files                 |
| `wessh-media` | ffmpeg subprocesses: webcam capture, virtual camera / stream output, video    |
| `wessh-app`   | the `wessh` binary: CLI, headless commands, installer and the egui app        |

Dependencies point one way: `wessh-app` → `wessh-media` → `wessh-core`. The core knows nothing
about ffmpeg, the CLI or the UI, and the media crate never runs inference.

## Benchmarks

`wessh bench` runs the full pipeline on one image and prints per-stage timings averaged over
`-n` iterations, after a warm-up:

```sh
cargo build --release
./target/release/wessh bench -s source.jpg -t target.jpg -n 200 --device cuda
./target/release/wessh bench -s source.jpg -t target.jpg -n 200 --device cpu --detector-size 320
```

When a change touches the pipeline, include before/after numbers (device, GPU/CPU model,
image size, options) in the pull request. For the README comparison with Deep-Live-Cam,
measure both tools on the same machine, with the same source and target images, the same swap
model and without the enhancer, and report the median per-frame latency.

## Releases

`.github/workflows/ci.yml` runs the checks above on every push. A push to main that bumps
`version` in the root `Cargo.toml` also builds Linux x86-64 / arm64, Windows x86-64 and macOS
arm64 / x86-64 archives with `scripts/package.sh` and, once the checks pass, publishes them with
a `SHA256SUMS` file as release `vX.Y.Z`. The release notes list the commit subjects since the
previous release (the app's update dialog shows them too), so write subjects for users.

`sources::ORT_VERSION` in `crates/wessh-app/src/install/sources.rs` pins the ONNX Runtime
release `wessh setup` installs; update it together with the `ort` crate.

## Releasing

1. Bump `version` in the workspace `Cargo.toml` (and run `cargo build` so `Cargo.lock` follows).
2. Commit ("Release version 0.3.0", left out of the notes) and push to main. Don't push a tag
   as well: the workflow creates it, and a second run would race the first.
3. The one-line installers and the in-app updater pick up the new release once it is published.

Pushing a tag with a suffix (`v0.3.0-rc.1`) publishes a pre-release, which the installers skip.
