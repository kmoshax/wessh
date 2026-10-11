# Contributing to weshy

Thanks for helping. weshy aims to be the fastest and easiest-to-install face swapper, so
changes are judged on two things above all: latency and install friction.

## Getting started

You need Rust 1.99 (pinned in `rust-toolchain.toml`, which rustup picks up), `curl`, and `ffmpeg`
on `PATH`.

```sh
git clone https://github.com/kmoshax/weshy && cd weshy
cargo build
# Runtime, CUDA libraries and models into a throwaway data dir, leaving your own install alone:
WESHY_HOME=/tmp/weshy-dev cargo run -- setup
WESHY_HOME=/tmp/weshy-dev cargo run -- doctor
WESHY_HOME=/tmp/weshy-dev cargo run --release            # the app
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

- Libraries (`weshy-core`, `weshy-media`) return typed errors (`thiserror`); the binary
  (`weshy-app`) uses `anyhow` and adds context.
- No magic numbers: name constants and say where they come from (a model's input size, a
  driver version, a protocol limit).
- No `unsafe` (the workspace forbids it).
- Keep the hot path allocation-free and avoid whole-frame copies; measure before and after
  (see below).
- Dependencies must earn their place: the release binary is a few megabytes and should stay that way.
  Prefer the standard library, then small well-maintained crates.

See [CLAUDE.md](CLAUDE.md) for the architecture and crate boundaries.

## Benchmarks

`weshy bench` runs the full pipeline on one image and prints per-stage timings averaged over
`-n` iterations, after a warm-up:

```sh
cargo build --release
./target/release/weshy bench -s source.jpg -t target.jpg -n 200 --device cuda
./target/release/weshy bench -s source.jpg -t target.jpg -n 200 --device cpu --detector-size 320
```

When a change touches the pipeline, include before/after numbers (device, GPU/CPU model,
image size, options) in the pull request. For the README comparison with Deep-Live-Cam,
measure both tools on the same machine, with the same source and target images, the same swap
model and without the enhancer, and report the median per-frame latency.

## Releases

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`, which builds Linux x86-64 /
arm64, Windows x86-64 and macOS arm64 / x86-64 archives with `scripts/package.sh`, attaches
them with a `SHA256SUMS` file to a draft GitHub release, and leaves it for review. Bump
`version` in the root `Cargo.toml` first.

`sources::ORT_VERSION` in `crates/weshy-app/src/commands/setup/sources.rs` pins the ONNX Runtime
release `weshy setup` installs; update it together with the `ort` crate.

## Releasing

1. Bump `version` in the workspace `Cargo.toml` (and run `cargo build` so `Cargo.lock` follows).
2. Commit, then tag and push: `git tag v0.3.0 && git push origin main v0.3.0`.
3. The Release workflow checks that the tag matches the version, runs the lints and tests,
   builds Linux (x86-64, arm64), Windows and macOS (arm64, x86-64) archives and publishes them
   with `SHA256SUMS`. The one-line installers pick up the new release immediately.

A tag with a suffix (`v0.3.0-rc.1`) is published as a pre-release, which the installers skip.
