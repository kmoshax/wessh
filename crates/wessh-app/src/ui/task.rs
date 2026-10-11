//! One-shot background work for the UI thread: probing files, listing devices, saving images.

use eframe::egui::Context;
use parking_lot::Mutex;
use std::sync::Arc;

/// A value computed on a helper thread and picked up on a later frame.
pub struct Task<T> {
    slot: Arc<Mutex<Option<T>>>,
    running: bool,
}

impl<T> Default for Task<T> {
    fn default() -> Self {
        Self { slot: Arc::new(Mutex::new(None)), running: false }
    }
}

impl<T: Send + 'static> Task<T> {
    /// Run `work` on a new thread, replacing any unfinished run (its result is dropped).
    pub fn spawn(&mut self, ctx: &Context, work: impl FnOnce() -> T + Send + 'static) {
        let slot = Arc::new(Mutex::new(None));
        self.slot = slot.clone();
        self.running = true;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            *slot.lock() = Some(work());
            ctx.request_repaint();
        });
    }

    /// The result, once, when the work has finished.
    pub fn poll(&mut self) -> Option<T> {
        let value = self.slot.lock().take();
        if value.is_some() {
            self.running = false;
        }
        value
    }

    /// Work was started and its result not yet collected.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }
}
