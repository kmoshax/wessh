use parking_lot::{Condvar, Mutex};
use std::time::Duration;
use wessh_core::Frame;

/// Frames kept for reuse: the one being filled, the one waiting and the one being consumed
/// cycle between producer and consumer without allocating.
const SPARES: usize = 2;

/// Single-slot mailbox that only keeps the newest frame, so a slow consumer never falls
/// behind the camera: latency stays bounded at one frame.
///
/// Frames are recycled: a frame replaced before it was taken, or handed back with
/// [`LatestFrame::recycle`], becomes a spare the producer can refill with
/// [`LatestFrame::spare`] — no per-frame allocation in the steady state.
#[derive(Debug, Default)]
pub struct LatestFrame {
    slot: Mutex<Slot>,
    ready: Condvar,
}

#[derive(Debug, Default)]
struct Slot {
    frame: Option<Frame>,
    sequence: u64,
    spares: Vec<Frame>,
}

impl Slot {
    fn keep(&mut self, frame: Frame) {
        if self.spares.len() < SPARES {
            self.spares.push(frame);
        }
    }
}

impl LatestFrame {
    pub fn publish(&self, frame: Frame) {
        let mut slot = self.slot.lock();
        if let Some(stale) = slot.frame.replace(frame) {
            slot.keep(stale);
        }
        slot.sequence += 1;
        self.ready.notify_all();
    }

    /// Wait up to `timeout` for a frame newer than `*seen`, updating `*seen`.
    pub fn take_newer(&self, seen: &mut u64, timeout: Duration) -> Option<Frame> {
        let mut slot = self.slot.lock();
        if slot.sequence == *seen {
            self.ready.wait_for(&mut slot, timeout);
        }
        if slot.sequence == *seen {
            return None;
        }
        *seen = slot.sequence;
        slot.frame.take()
    }

    /// Hand a consumed frame back for the producer to refill.
    pub fn recycle(&self, frame: Frame) {
        self.slot.lock().keep(frame);
    }

    /// A `width`×`height` frame to fill: a recycled one when available (its old pixels are
    /// still there), else a new one.
    #[must_use]
    pub fn spare(&self, width: usize, height: usize) -> Frame {
        let mut slot = self.slot.lock();
        slot.spares.retain(|f| (f.width, f.height) == (width, height));
        slot.spares.pop().unwrap_or_else(|| Frame::new(width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_the_newest_frame() {
        let mailbox = LatestFrame::default();
        mailbox.publish(Frame::new(1, 1));
        mailbox.publish(Frame::new(2, 2));
        let mut seen = 0;
        let frame = mailbox.take_newer(&mut seen, Duration::ZERO).expect("frame");
        assert_eq!((frame.width, seen), (2, 2));
        assert!(mailbox.take_newer(&mut seen, Duration::from_millis(1)).is_none());
    }

    #[test]
    fn recycles_replaced_and_returned_frames() {
        let mailbox = LatestFrame::default();
        let mut first = mailbox.spare(4, 2);
        first.data[0] = 7;
        mailbox.publish(first);
        mailbox.publish(mailbox.spare(4, 2)); // replaces the first, which becomes a spare
        assert_eq!(mailbox.spare(4, 2).data[0], 7);
        let mut seen = 0;
        let taken = mailbox.take_newer(&mut seen, Duration::ZERO).expect("frame");
        mailbox.recycle(taken);
        // Spares of another size are dropped.
        assert_eq!(mailbox.spare(8, 8).width, 8);
        assert_eq!(mailbox.slot.lock().spares.len(), 0);
    }
}
