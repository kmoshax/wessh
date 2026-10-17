//! A mapped view of the frames file ([`crate::layout`]), shared by the writer (wessh) and the
//! readers (the media source in the frame server and in apps).

use crate::layout::{HEADER_LEN, Header, LATEST, WRITING};
use std::fs::File;
use std::os::windows::io::AsRawHandle;
use std::sync::atomic::AtomicU64;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
    UnmapViewOfFile,
};
use windows_core::PCWSTR;

/// A whole-file view, unmapped on drop. The file handle may be closed once mapped.
pub struct Mapping {
    section: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

// SAFETY: the view is plain memory owned by this value (no thread affinity); concurrent access
// goes through atomics or is checked for tearing by the protocol in `layout`.
unsafe impl Send for Mapping {}

impl Mapping {
    /// Map the first `len` bytes of `file` (which must be at least that long).
    pub fn new(file: &File, len: usize, writable: bool) -> windows_core::Result<Self> {
        let (protect, access) = if writable { (PAGE_READWRITE, FILE_MAP_WRITE) } else { (PAGE_READONLY, FILE_MAP_READ) };
        let handle = HANDLE(file.as_raw_handle());
        // SAFETY: `handle` is a valid open file handle borrowed for the call; size 0 maps the
        // file's current size, and an unnamed section needs no security attributes.
        let section = unsafe { CreateFileMappingW(handle, None, protect, 0, 0, PCWSTR::null())? };
        // SAFETY: `section` was just created; `len` is no more than the file size (checked by
        // the callers), so the view covers only existing pages.
        let view = unsafe { MapViewOfFile(section, access, 0, 0, len) };
        if view.Value.is_null() {
            let error = windows_core::Error::from_thread();
            // SAFETY: `section` is a handle we own and no longer need.
            let _ = unsafe { CloseHandle(section) };
            return Err(error);
        }
        Ok(Self { section, view, len })
    }

    /// Mapped bytes (readers map the whole file, which may be larger than one stream needs).
    #[cfg(feature = "source")]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// The header bytes, copied out (the writer may change them at any time).
    pub fn header_bytes(&self) -> [u8; HEADER_LEN] {
        let mut bytes = [0u8; HEADER_LEN];
        // SAFETY: every mapping is at least `HEADER_LEN` long (checked by the callers) and the
        // view is valid for reads while `self` lives; volatile because another process writes it.
        unsafe {
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = self.base().add(i).read_volatile();
            }
        }
        bytes
    }

    /// Write the header's fixed fields (writer only, before any frame).
    pub fn write_header(&mut self, header: Header) {
        let mut bytes = self.header_bytes();
        header.write(&mut bytes);
        // SAFETY: as in `header_bytes`; the writer mapping is writable.
        unsafe {
            for (i, byte) in bytes.iter().enumerate().take(LATEST) {
                self.base().add(i).write_volatile(*byte);
            }
        }
    }

    /// The `latest` frame counter.
    pub fn latest(&self) -> &AtomicU64 {
        self.counter(LATEST)
    }

    /// The `writing` frame counter.
    pub fn writing(&self) -> &AtomicU64 {
        self.counter(WRITING)
    }

    fn counter(&self, offset: usize) -> &AtomicU64 {
        // SAFETY: `offset` is one of the 8-aligned header fields inside `HEADER_LEN`; the view is
        // page-aligned and lives as long as `&self`, and the field is only ever accessed atomically.
        unsafe { AtomicU64::from_ptr(self.base().add(offset).cast()) }
    }

    /// `range` of the view, or `None` if it lies outside.
    #[cfg(feature = "source")]
    pub fn bytes(&self, range: std::ops::Range<usize>) -> Option<&[u8]> {
        (range.end <= self.len).then(|| {
            // SAFETY: in bounds and alive for `&self`. Another process may write these bytes
            // meanwhile; readers detect that with the frame counters and discard the copy.
            unsafe { std::slice::from_raw_parts(self.base().add(range.start), range.len()) }
        })
    }

    /// `range` of the view for writing, or `None` if it lies outside.
    pub fn bytes_mut(&mut self, range: std::ops::Range<usize>) -> Option<&mut [u8]> {
        (range.end <= self.len).then(|| {
            // SAFETY: in bounds, writable (writer mapping) and exclusively borrowed. Readers only
            // read the slot being written when they are behind, and then discard their copy.
            unsafe { std::slice::from_raw_parts_mut(self.base().add(range.start), range.len()) }
        })
    }

    fn base(&self) -> *mut u8 {
        self.view.Value.cast()
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the view and the section were created by `new` and are released exactly once;
        // no borrow of the view outlives `self`.
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            let _ = CloseHandle(self.section);
        }
    }
}
