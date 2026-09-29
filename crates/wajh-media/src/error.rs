pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not start {program} (is it installed and on PATH?): {source}")]
    Spawn { program: String, source: std::io::Error },

    #[error("ffmpeg exited with {0}")]
    Exited(std::process::ExitStatus),

    #[error("no video stream in {0}")]
    NoVideoStream(String),

    #[error("frame is {got:?} but the output was opened as {expected:?}")]
    FrameSize { expected: (usize, usize), got: (usize, usize) },

    #[error("{0}")]
    Unsupported(&'static str),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
