//! The presenter thread: takes processed frames off the inference thread and does the rest —
//! RGBA conversion for the preview and the write to the virtual camera — so inference can start
//! on the next camera frame straight away.
//!
//! Frames arrive through a one-slot [`LatestFrame`] mailbox, so if presenting ever falls behind,
//! stale frames are dropped rather than queued. Buffers go round in circles: presented frames
//! return to the camera's mailbox for refilling, and preview pixel buffers come back from the
//! UI (see [`super::LiveSession::recycle_preview`]).

use super::state::{Image, Shared};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;
use wessh_core::Frame;
use wessh_media::{FrameSink, LatestFrame};

/// How long the presenter waits for a frame before checking for control messages.
const POLL: Duration = Duration::from_millis(100);

/// What the inference thread tells the presenter besides frames.
pub(super) enum Control {
    /// Where to send frames from now on: the virtual camera, or nowhere. Also names the
    /// camera mailbox that presented frames are recycled into.
    Stream { output: Option<FrameSink>, camera: Option<Arc<LatestFrame>> },
}

/// Handle owned by the inference thread. Dropping it stops the presenter thread.
pub(super) struct Presenter {
    frames: Arc<LatestFrame>,
    control: Option<Sender<Control>>,
    thread: Option<JoinHandle<()>>,
}

impl Presenter {
    pub fn spawn(shared: Arc<Shared>, repaint: Arc<dyn Fn() + Send + Sync>) -> std::io::Result<Self> {
        let frames = Arc::new(LatestFrame::default());
        let (control, inbox) = crossbeam_channel::unbounded();
        let mailbox = frames.clone();
        let thread = std::thread::Builder::new().name("present".into()).spawn(move || run(&mailbox, &inbox, &shared, &*repaint))?;
        Ok(Self { frames, control: Some(control), thread: Some(thread) })
    }

    /// Present `frame` (replacing a previous one that hasn't been presented yet).
    pub fn present(&self, frame: Frame) {
        self.frames.publish(frame);
    }

    pub fn control(&self, message: Control) {
        if let Some(control) = &self.control {
            let _ = control.send(message);
        }
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.control = None; // disconnects the presenter's inbox
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(frames: &LatestFrame, inbox: &Receiver<Control>, shared: &Shared, repaint: &dyn Fn()) {
    let (mut output, mut camera): (Option<FrameSink>, Option<Arc<LatestFrame>>) = (None, None);
    let mut seen = 0;
    loop {
        loop {
            match inbox.try_recv() {
                Ok(Control::Stream { output: o, camera: c }) => (output, camera) = (o, c),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let Some(frame) = frames.take_newer(&mut seen, POLL) else { continue };
        if let Some(sink) = output.as_mut()
            && let Err(e) = sink.write(&frame)
        {
            output = None;
            shared.update(|s| {
                s.output_active = false;
                s.notice = Some(format!("virtual camera stopped: {e}"));
            });
        }
        publish_preview(shared, &frame);
        repaint();
        match &camera {
            Some(camera) => camera.recycle(frame),
            None => frames.recycle(frame),
        }
    }
}

/// Convert `frame` for display into a recycled buffer: the preview the UI hasn't taken yet,
/// one it handed back, or — at the start — a new one.
fn publish_preview(shared: &Shared, frame: &Frame) {
    let mut rgba = shared.preview.lock().take().map(|image| image.rgba).or_else(|| shared.recycled.lock().take()).unwrap_or_default();
    frame.write_rgba(&mut rgba);
    *shared.preview.lock() = Some(Image { width: frame.width, height: frame.height, rgba });
}
