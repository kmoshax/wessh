//! The file the wessh process and the media source share frames through.
//!
//! The camera frame server runs as Local Service in session 0, and apps load the source too, so
//! the frames cannot live in a session-local named section, and a global one needs a privilege
//! users lack. A file mapping needs neither: wessh creates the file under `%ProgramData%\wessh`,
//! which every account can read, and both sides map it. Nothing is written to disk while memory
//! is available (temporary file attribute), so it is plain shared memory.
//!
//! Layout: a [`HEADER_LEN`]-byte header, then [`SLOTS`] NV12 frames. Frame `n` (counting from 1)
//! goes into slot `n % SLOTS`. The writer stores `writing = n` before filling a slot and
//! `latest = n` after; a reader copies slot `latest % SLOTS` and then checks `writing`: the copy
//! is whole unless the writer has since started frame `latest + SLOTS`, which reuses that slot.
//! No locks, so neither side ever waits for the other.
//!
//! The reader runs in the frame server with data a user process wrote, so it validates every
//! field and bounds-checks every access ([`Header::parse`], [`frame_range`]).

/// `WVCM`: marks a frames file.
pub const MAGIC: u32 = u32::from_le_bytes(*b"WVCM");
/// Bumped whenever the header, the pixel format or the install layout change; the installed
/// source records it in the registry so the app can ask for a reinstall.
pub const PROTOCOL: u32 = 1;
/// Bytes before the first frame (room to grow; the fields end at [`WRITING`] + 8).
pub const HEADER_LEN: usize = 64;
/// Frames in flight: one being written, one being read, one spare so neither waits.
pub const SLOTS: u64 = 3;
/// Largest side accepted from the file (8K), which bounds what the reader will map.
pub const MAX_SIDE: u32 = 8192;
/// Frame rates the media type may advertise.
pub const FPS_RANGE: std::ops::RangeInclusive<u32> = 1..=240;

/// Header field offsets (all little-endian, naturally aligned).
pub const OFFSET_MAGIC: usize = 0;
pub const OFFSET_PROTOCOL: usize = 4;
pub const OFFSET_WIDTH: usize = 8;
pub const OFFSET_HEIGHT: usize = 12;
pub const OFFSET_FPS: usize = 16;
/// `u64`: the last complete frame, 0 before the first.
pub const LATEST: usize = 24;
/// `u64`: the frame being written (equal to `latest` when idle).
pub const WRITING: usize = 32;

/// NV12 luma value of black (BT.601 studio swing) and the neutral chroma value.
pub const BLACK_LUMA: u8 = 16;
pub const NEUTRAL_CHROMA: u8 = 128;

/// What the header says about the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Header {
    /// The header fields if `bytes` start with a valid header for this protocol.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let field = |offset: usize| bytes.get(offset..offset + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        if field(OFFSET_MAGIC)? != MAGIC || field(OFFSET_PROTOCOL)? != PROTOCOL {
            return None;
        }
        let header = Self { width: field(OFFSET_WIDTH)?, height: field(OFFSET_HEIGHT)?, fps: field(OFFSET_FPS)? };
        header.is_valid().then_some(header)
    }

    /// Write the fixed fields (not the frame counters) into `bytes[..HEADER_LEN]`.
    pub fn write(self, bytes: &mut [u8]) {
        for (offset, value) in [
            (OFFSET_MAGIC, MAGIC),
            (OFFSET_PROTOCOL, PROTOCOL),
            (OFFSET_WIDTH, self.width),
            (OFFSET_HEIGHT, self.height),
            (OFFSET_FPS, self.fps),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Even sides (NV12 subsamples chroma 2×2) within bounds, and a sane frame rate.
    #[must_use]
    pub fn is_valid(self) -> bool {
        let side = |v: u32| (2..=MAX_SIDE).contains(&v) && v.is_multiple_of(2);
        side(self.width) && side(self.height) && FPS_RANGE.contains(&self.fps)
    }

    /// Bytes of one NV12 frame: full-resolution luma, then interleaved chroma at quarter size.
    #[must_use]
    pub const fn frame_len(self) -> usize {
        let pixels = self.width as usize * self.height as usize;
        pixels + pixels / 2
    }

    /// Size of the whole file.
    #[must_use]
    pub const fn file_len(self) -> usize {
        HEADER_LEN + SLOTS as usize * self.frame_len()
    }
}

/// Byte range of frame `n`'s slot within the file.
#[must_use]
pub const fn frame_range(header: Header, n: u64) -> std::ops::Range<usize> {
    let start = HEADER_LEN + (n % SLOTS) as usize * header.frame_len();
    start..start + header.frame_len()
}

/// Whether a copy of frame `read` may have been overwritten: the writer had started a frame
/// that reuses the same slot by the time the copy finished.
#[must_use]
pub const fn is_torn(read: u64, writing_after_copy: u64) -> bool {
    writing_after_copy >= read + SLOTS
}

/// Copy an NV12 frame into a buffer whose rows are `pitch` bytes apart (Media Foundation pads
/// rows to its alignment). Returns `false`, copying nothing, if either side is too small.
#[must_use]
pub fn copy_nv12(src: &[u8], header: Header, dst: &mut [u8], pitch: usize) -> bool {
    let (width, height) = (header.width as usize, header.height as usize);
    let rows = height + height / 2;
    if pitch < width || src.len() < header.frame_len() || dst.len() < pitch * (rows - 1) + width {
        return false;
    }
    for (row, line) in src.chunks_exact(width).take(rows).enumerate() {
        dst[row * pitch..row * pitch + width].copy_from_slice(line);
    }
    true
}

/// Fill an NV12 buffer (rows `pitch` bytes apart) with black. Returns `false` if it is too small.
#[must_use]
pub fn fill_black(header: Header, dst: &mut [u8], pitch: usize) -> bool {
    let (width, height) = (header.width as usize, header.height as usize);
    let rows = height + height / 2;
    if pitch < width || dst.len() < pitch * (rows - 1) + width {
        return false;
    }
    for row in 0..rows {
        let value = if row < height { BLACK_LUMA } else { NEUTRAL_CHROMA };
        dst[row * pitch..row * pitch + width].fill(value);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const HD: Header = Header { width: 1280, height: 720, fps: 30 };

    #[test]
    fn header_round_trips_and_rejects_bad_values() {
        let mut bytes = [0u8; HEADER_LEN];
        HD.write(&mut bytes);
        assert_eq!(Header::parse(&bytes), Some(HD));
        assert_eq!(Header::parse(&bytes[..8]), None, "truncated");
        for bad in [Header { width: 641, ..HD }, Header { height: 0, ..HD }, Header { width: MAX_SIDE + 2, ..HD }, Header { fps: 0, ..HD }]
        {
            bad.write(&mut bytes);
            assert_eq!(Header::parse(&bytes), None, "{bad:?}");
        }
        HD.write(&mut bytes);
        bytes[OFFSET_PROTOCOL] ^= 1;
        assert_eq!(Header::parse(&bytes), None, "other protocol");
    }

    #[test]
    fn slots_follow_the_frame_number() {
        let len = HD.frame_len();
        assert_eq!(len, 1280 * 720 * 3 / 2);
        assert_eq!(frame_range(HD, 3), HEADER_LEN..HEADER_LEN + len);
        assert_eq!(frame_range(HD, 4).start, HEADER_LEN + len);
        assert_eq!(frame_range(HD, 5).end, HD.file_len());
    }

    #[test]
    fn a_copy_is_torn_only_once_its_slot_is_rewritten() {
        assert!(!is_torn(7, 7));
        assert!(!is_torn(7, 9), "the writer is filling the two other slots");
        assert!(is_torn(7, 10), "frame 10 reuses frame 7's slot");
    }

    #[test]
    fn copies_and_fills_respect_the_pitch() {
        let small = Header { width: 2, height: 2, fps: 30 };
        let src = [1, 2, 3, 4, 5, 6];
        let mut dst = [0u8; 12];
        assert!(copy_nv12(&src, small, &mut dst, 4));
        assert_eq!(dst, [1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0]);
        assert!(fill_black(small, &mut dst, 4));
        assert_eq!(dst, [16, 16, 0, 0, 16, 16, 0, 0, 128, 128, 0, 0]);
        assert!(!copy_nv12(&src, small, &mut dst[..9], 4), "destination too short");
        assert!(!copy_nv12(&src[..5], small, &mut dst, 4), "source too short");
        assert!(!fill_black(small, &mut dst, 1), "pitch narrower than a row");
    }
}
