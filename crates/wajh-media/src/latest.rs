use parking_lot::{Condvar, Mutex};
use std::time::Duration;
use wajh_core::Frame;

/// Single-slot mailbox that only keeps the newest frame, so a slow consumer never falls
/// behind the camera: latency stays bounded at one frame.
#[derive(Debug, Default)]
pub struct LatestFrame {
    slot: Mutex<Slot>,
    ready: Condvar,
}

#[derive(Debug, Default)]
struct Slot {
    frame: Option<Frame>,
    sequence: u64,
}

impl LatestFrame {
    pub fn publish(&self, frame: Frame) {
        let mut slot = self.slot.lock();
        slot.frame = Some(frame);
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
}
