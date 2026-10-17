//! The media source the Windows camera frame server loads from `wessh-vcam.dll`.
//!
//! COM shape (as Windows' virtual camera sample and the frame server expect): the registered
//! class is an `IMFActivate`, whose `ActivateObject` hands out one media source
//! (`IMFMediaSourceEx` + `IKsControl` + `IMFGetService`) with one video stream
//! (`IMFMediaStream2` + `IKsControl`). Every object is also an `IMFAttributes` store.
//!
//! The stream advertises one NV12 type at the size and rate wessh wrote into the frames file
//! ([`crate::layout`]) when the source was created. Each `RequestSample` copies the newest whole
//! frame into a pooled sample, or black while there is none. This code runs inside the frame
//! server and other apps: it never panics on purpose and checks everything it reads.

#![allow(clippy::inline_always, clippy::ref_as_ptr, reason = "in the code windows-rs' #[implement] generates")]

use crate::layout::{self, HEADER_LEN, Header};
use crate::shared::Mapping;
use std::ffi::c_void;
use std::fs::OpenOptions;
use std::mem::ManuallyDrop;
use std::os::windows::fs::OpenOptionsExt;
use std::sync::atomic::{Ordering, fence};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{
    CLASS_E_CLASSNOTAVAILABLE, CLASS_E_NOAGGREGATION, E_INVALIDARG, E_POINTER, E_UNEXPECTED, ERROR_SET_NOT_FOUND, S_FALSE, S_OK,
};
use windows::Win32::Media::KernelStreaming::{
    IKsControl, IKsControl_Impl, KSCAMERAPROFILE_HighFrameRate, KSCAMERAPROFILE_Legacy, KSIDENTIFIER, PINNAME_VIDEO_CAPTURE,
};
use windows::Win32::Media::MediaFoundation::{
    IMF2DBuffer2, IMFActivate, IMFActivate_Impl, IMFAsyncCallback, IMFAsyncResult, IMFAttributes, IMFAttributes_Impl, IMFGetService,
    IMFGetService_Impl, IMFMediaBuffer, IMFMediaEvent, IMFMediaEventGenerator_Impl, IMFMediaEventQueue, IMFMediaSource,
    IMFMediaSource_Impl, IMFMediaSourceEx, IMFMediaSourceEx_Impl, IMFMediaStream_Impl, IMFMediaStream2, IMFMediaStream2_Impl, IMFMediaType,
    IMFPresentationDescriptor, IMFStreamDescriptor, IMFVideoSampleAllocatorEx, MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS, MEMediaSample,
    MENewStream, MESourceStarted, MESourceStopped, MEStreamStarted, MEStreamStopped, MEUpdatedStream, MF_ATTRIBUTE_TYPE,
    MF_ATTRIBUTES_MATCH_TYPE, MF_DEVICEMFT_SENSORPROFILE_COLLECTION, MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES,
    MF_DEVICESTREAM_FRAMESERVER_SHARED, MF_DEVICESTREAM_STREAM_CATEGORY, MF_DEVICESTREAM_STREAM_ID, MF_E_INVALID_STATE_TRANSITION,
    MF_E_INVALIDSTREAMNUMBER, MF_E_SHUTDOWN, MF_E_UNSUPPORTED_SERVICE, MF_MT_ALL_SAMPLES_INDEPENDENT, MF_MT_AVG_BITRATE,
    MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_PIXEL_ASPECT_RATIO,
    MF_MT_SUBTYPE, MF_STREAM_STATE, MF_STREAM_STATE_RUNNING, MF_STREAM_STATE_STOPPED, MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES,
    MF2DBuffer_LockFlags_Write, MFCreateAttributes, MFCreateEventQueue, MFCreateMediaType, MFCreatePresentationDescriptor,
    MFCreateSensorProfile, MFCreateSensorProfileCollection, MFCreateStreamDescriptor, MFCreateVideoSampleAllocatorEx,
    MFFrameSourceTypes_Color, MFGetSystemTime, MFMEDIASOURCE_IS_LIVE, MFMediaType_Video, MFSampleExtension_Token,
    MFT_TRANSFORM_CLSID_Attribute, MFVideoFormat_NV12, MFVideoInterlace_Progressive,
};
use windows::Win32::Storage::FileSystem::{FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE};
use windows::Win32::System::Com::StructuredStorage::{PROPVARIANT, PROPVARIANT_0, PROPVARIANT_0_0, PROPVARIANT_0_0_0};
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl};
use windows::Win32::System::Variant::VT_I8;
use windows_core::{BOOL, ComObject, GUID, HRESULT, IUnknown, Interface, PCWSTR, PWSTR, Ref, Result, implement, w};

/// Advertised when wessh has not written a frames file (the camera then shows black).
const FALLBACK: Header = Header { width: 1280, height: 720, fps: 30 };
/// The only stream's identifier.
const STREAM_ID: u32 = 0;
/// Samples the allocator keeps in its pool (the frame server holds a few at a time).
const SAMPLE_POOL: u32 = 10;
/// Media Foundation time unit: 100 ns.
const TICKS_PER_SECOND: i64 = 10_000_000;
/// Copies attempted before a frame the writer keeps overwriting is given up for black.
const READ_ATTEMPTS: usize = 3;
/// How often a missing frames file is looked for again.
const REOPEN_INTERVAL: Duration = Duration::from_secs(1);
/// Sensor profiles (filter strings from Windows' virtual camera sample): the legacy profile for
/// rates up to 30 fps, high frame rate above, so the one media type always matches one.
const LEGACY_PROFILE_FILTER: PCWSTR = w!("((RES==;FRT<=30,1;SUT==))");
const HIGH_FRAME_RATE_PROFILE_FILTER: PCWSTR = w!("((RES==;FRT>=60,1;SUT==))");

// ── DLL exports ─────────────────────────────────────────────────────────────────────────────

/// COM entry point: the class factory for [`crate::CLSID`].
///
/// # Safety
///
/// Called by COM with valid (or null, which is rejected) pointers.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DllGetClassObject(class: *const GUID, iid: *const GUID, object: *mut *mut c_void) -> HRESULT {
    if class.is_null() || iid.is_null() || object.is_null() {
        return E_POINTER;
    }
    // SAFETY: checked non-null above; COM passes a writable out pointer and readable GUIDs.
    unsafe {
        object.write(std::ptr::null_mut());
        if class.read() != GUID::from_u128(crate::CLSID) {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        let factory: IClassFactory = Factory.into();
        factory.query(iid, object)
    }
}

/// COM asks whether the DLL may be unloaded: never, since an object may still be alive and
/// staying loaded costs nothing.
#[unsafe(no_mangle)]
pub extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

// ── Helpers ─────────────────────────────────────────────────────────────────────────────────

/// Lock, ignoring poisoning (a panic aborts the process anyway).
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn new_attributes() -> Result<IMFAttributes> {
    /// Initial capacity; the store grows as needed.
    const CAPACITY: u32 = 8;
    let mut attributes = None;
    // SAFETY: a valid out pointer for the new store.
    unsafe { MFCreateAttributes(&raw mut attributes, CAPACITY)? };
    attributes.ok_or_else(|| E_UNEXPECTED.into())
}

/// A `VT_I8` property value (event timestamps).
fn time_value(ticks: i64) -> PROPVARIANT {
    PROPVARIANT {
        Anonymous: PROPVARIANT_0 {
            Anonymous: ManuallyDrop::new(PROPVARIANT_0_0 {
                vt: VT_I8,
                wReserved1: 0,
                wReserved2: 0,
                wReserved3: 0,
                Anonymous: PROPVARIANT_0_0_0 { hVal: ticks },
            }),
        },
    }
}

/// Two 32-bit values packed the way `MFSetAttributeSize` / `MFSetAttributeRatio` do.
const fn packed(high: u32, low: u32) -> u64 {
    ((high as u64) << 32) | low as u64
}

fn now() -> i64 {
    // SAFETY: no arguments; reads Media Foundation's clock.
    unsafe { MFGetSystemTime() }
}

/// Open the frames file wessh writes, if it exists and is readable.
fn open_frames() -> Option<Mapping> {
    let file = OpenOptions::new()
        .read(true)
        .share_mode((FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE).0)
        .open(crate::frames_path())
        .ok()?;
    let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
    if len < HEADER_LEN {
        return None;
    }
    Mapping::new(&file, len, false).ok()
}

/// Forward every `IMFAttributes` method of `$outer` to its `attributes` field unchanged.
macro_rules! delegate_attributes {
    ($outer:ty) => {
        impl IMFAttributes_Impl for $outer {
            fn GetItem(&self, key: *const GUID, value: *mut PROPVARIANT) -> Result<()> {
                forward!(self, GetItem, key, value)
            }
            fn GetItemType(&self, key: *const GUID) -> Result<MF_ATTRIBUTE_TYPE> {
                let mut kind = MF_ATTRIBUTE_TYPE::default();
                forward!(self, GetItemType, key, &raw mut kind).map(|()| kind)
            }
            fn CompareItem(&self, key: *const GUID, value: *const PROPVARIANT) -> Result<BOOL> {
                let mut equal = BOOL::default();
                forward!(self, CompareItem, key, value, &raw mut equal).map(|()| equal)
            }
            fn Compare(&self, theirs: Ref<IMFAttributes>, kind: MF_ATTRIBUTES_MATCH_TYPE) -> Result<BOOL> {
                let mut equal = BOOL::default();
                forward!(self, Compare, raw(&theirs), kind, &raw mut equal).map(|()| equal)
            }
            fn GetUINT32(&self, key: *const GUID) -> Result<u32> {
                let mut value = 0;
                forward!(self, GetUINT32, key, &raw mut value).map(|()| value)
            }
            fn GetUINT64(&self, key: *const GUID) -> Result<u64> {
                let mut value = 0;
                forward!(self, GetUINT64, key, &raw mut value).map(|()| value)
            }
            fn GetDouble(&self, key: *const GUID) -> Result<f64> {
                let mut value = 0.0;
                forward!(self, GetDouble, key, &raw mut value).map(|()| value)
            }
            fn GetGUID(&self, key: *const GUID) -> Result<GUID> {
                let mut value = GUID::zeroed();
                forward!(self, GetGUID, key, &raw mut value).map(|()| value)
            }
            fn GetStringLength(&self, key: *const GUID) -> Result<u32> {
                let mut value = 0;
                forward!(self, GetStringLength, key, &raw mut value).map(|()| value)
            }
            fn GetString(&self, key: *const GUID, value: PWSTR, capacity: u32, length: *mut u32) -> Result<()> {
                forward!(self, GetString, key, value, capacity, length)
            }
            fn GetAllocatedString(&self, key: *const GUID, value: *mut PWSTR, length: *mut u32) -> Result<()> {
                forward!(self, GetAllocatedString, key, value, length)
            }
            fn GetBlobSize(&self, key: *const GUID) -> Result<u32> {
                let mut value = 0;
                forward!(self, GetBlobSize, key, &raw mut value).map(|()| value)
            }
            fn GetBlob(&self, key: *const GUID, buffer: *mut u8, capacity: u32, size: *mut u32) -> Result<()> {
                forward!(self, GetBlob, key, buffer, capacity, size)
            }
            fn GetAllocatedBlob(&self, key: *const GUID, buffer: *mut *mut u8, size: *mut u32) -> Result<()> {
                forward!(self, GetAllocatedBlob, key, buffer, size)
            }
            fn GetUnknown(&self, key: *const GUID, iid: *const GUID, object: *mut *mut c_void) -> Result<()> {
                forward!(self, GetUnknown, key, iid, object)
            }
            fn SetItem(&self, key: *const GUID, value: *const PROPVARIANT) -> Result<()> {
                forward!(self, SetItem, key, value)
            }
            fn DeleteItem(&self, key: *const GUID) -> Result<()> {
                forward!(self, DeleteItem, key)
            }
            fn DeleteAllItems(&self) -> Result<()> {
                forward!(self, DeleteAllItems)
            }
            fn SetUINT32(&self, key: *const GUID, value: u32) -> Result<()> {
                forward!(self, SetUINT32, key, value)
            }
            fn SetUINT64(&self, key: *const GUID, value: u64) -> Result<()> {
                forward!(self, SetUINT64, key, value)
            }
            fn SetDouble(&self, key: *const GUID, value: f64) -> Result<()> {
                forward!(self, SetDouble, key, value)
            }
            fn SetGUID(&self, key: *const GUID, value: *const GUID) -> Result<()> {
                forward!(self, SetGUID, key, value)
            }
            fn SetString(&self, key: *const GUID, value: &PCWSTR) -> Result<()> {
                forward!(self, SetString, key, *value)
            }
            fn SetBlob(&self, key: *const GUID, buffer: *const u8, size: u32) -> Result<()> {
                forward!(self, SetBlob, key, buffer, size)
            }
            fn SetUnknown(&self, key: *const GUID, value: Ref<IUnknown>) -> Result<()> {
                forward!(self, SetUnknown, key, raw(&value))
            }
            fn LockStore(&self) -> Result<()> {
                forward!(self, LockStore)
            }
            fn UnlockStore(&self) -> Result<()> {
                forward!(self, UnlockStore)
            }
            fn GetCount(&self) -> Result<u32> {
                let mut value = 0;
                forward!(self, GetCount, &raw mut value).map(|()| value)
            }
            fn GetItemByIndex(&self, index: u32, key: *mut GUID, value: *mut PROPVARIANT) -> Result<()> {
                forward!(self, GetItemByIndex, index, key, value)
            }
            fn CopyAllItems(&self, destination: Ref<IMFAttributes>) -> Result<()> {
                forward!(self, CopyAllItems, raw(&destination))
            }
        }
    };
}

/// Call `$method` of `$self.attributes` through its vtable with the arguments as given.
macro_rules! forward {
    ($self:ident, $method:ident $(, $argument:expr)*) => {{
        let attributes = &$self.attributes;
        // SAFETY: the caller's arguments are passed unchanged to the same method of a valid
        // IMFAttributes, so they meet its contract exactly as they met ours.
        unsafe { (Interface::vtable(attributes).$method)(Interface::as_raw(attributes) $(, $argument)*) }.ok()
    }};
}

/// The raw pointer behind an optional interface argument (null when absent).
fn raw<T: Interface>(value: &Ref<T>) -> *mut c_void {
    value.as_ref().map_or(std::ptr::null_mut(), Interface::as_raw)
}

/// What every `IKsControl` call answers: no property sets, methods or events.
fn no_ks_support() -> Result<()> {
    Err(ERROR_SET_NOT_FOUND.to_hresult().into())
}

// ── Class factory ───────────────────────────────────────────────────────────────────────────

#[implement(IClassFactory)]
struct Factory;

impl IClassFactory_Impl for Factory_Impl {
    fn CreateInstance(&self, outer: Ref<IUnknown>, iid: *const GUID, object: *mut *mut c_void) -> Result<()> {
        if object.is_null() || iid.is_null() {
            return Err(E_POINTER.into());
        }
        // SAFETY: checked non-null; COM passes a writable out pointer.
        unsafe { object.write(std::ptr::null_mut()) };
        if !outer.is_null() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        let activator: IMFActivate = Activator::new()?.into();
        // SAFETY: valid IID and out pointer (checked above).
        unsafe { activator.query(iid, object) }.ok()
    }

    fn LockServer(&self, _lock: BOOL) -> Result<()> {
        Ok(())
    }
}

// ── Activator ───────────────────────────────────────────────────────────────────────────────

#[implement(IMFActivate)]
struct Activator {
    attributes: IMFAttributes,
    source: Mutex<Option<ComObject<Source>>>,
}

impl Activator {
    fn new() -> Result<Self> {
        let attributes = new_attributes()?;
        // SAFETY: setting plain values on a valid attribute store.
        unsafe {
            attributes.SetUINT32(&MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES, 1)?;
            attributes.SetGUID(&MFT_TRANSFORM_CLSID_Attribute, &GUID::from_u128(crate::CLSID))?;
        }
        Ok(Self { attributes, source: Mutex::new(None) })
    }
}

delegate_attributes!(Activator_Impl);

impl IMFActivate_Impl for Activator_Impl {
    fn ActivateObject(&self, iid: *const GUID, object: *mut *mut c_void) -> Result<()> {
        if iid.is_null() || object.is_null() {
            return Err(E_POINTER.into());
        }
        let mut source = lock(&self.source);
        let source = match &*source {
            Some(existing) => existing.clone(),
            None => source.insert(Source::new(&self.attributes)?).clone(),
        };
        let unknown: IUnknown = source.to_interface();
        // SAFETY: valid IID and out pointer (checked above).
        unsafe { unknown.query(iid, object) }.ok()
    }

    fn ShutdownObject(&self) -> Result<()> {
        Ok(())
    }

    fn DetachObject(&self) -> Result<()> {
        lock(&self.source).take();
        Ok(())
    }
}

// ── Media source ────────────────────────────────────────────────────────────────────────────

#[implement(IMFMediaSourceEx, IMFGetService, IKsControl, IMFAttributes)]
struct Source {
    attributes: IMFAttributes,
    state: Mutex<SourceState>,
}

/// Emptied by `Shutdown`, after which every call fails with `MF_E_SHUTDOWN`.
#[derive(Default)]
struct SourceState {
    queue: Option<IMFEventQueue>,
    descriptor: Option<IMFPresentationDescriptor>,
    stream: Option<ComObject<Stream>>,
}

type IMFEventQueue = IMFMediaEventQueue;

impl Source {
    fn new(activator: &IMFAttributes) -> Result<ComObject<Self>> {
        let attributes = new_attributes()?;
        // SAFETY: valid attribute stores and freshly created profile objects throughout.
        unsafe {
            activator.CopyAllItems(&attributes)?;
            let profiles = MFCreateSensorProfileCollection()?;
            for (kind, filter) in
                [(KSCAMERAPROFILE_Legacy, LEGACY_PROFILE_FILTER), (KSCAMERAPROFILE_HighFrameRate, HIGH_FRAME_RATE_PROFILE_FILTER)]
            {
                let profile = MFCreateSensorProfile(&raw const kind, 0, PCWSTR::null())?;
                profile.AddProfileFilter(STREAM_ID, filter)?;
                profiles.AddProfile(&profile)?;
            }
            attributes.SetUnknown(&MF_DEVICEMFT_SENSORPROFILE_COLLECTION, &profiles)?;
        }
        let header = open_frames().and_then(|frames| Header::parse(&frames.header_bytes())).unwrap_or(FALLBACK);
        let source = ComObject::new(Self { attributes, state: Mutex::new(SourceState::default()) });
        let stream = Stream::new(source.to_interface::<IMFMediaSourceEx>().cast()?, header)?;
        // SAFETY: creating Media Foundation objects from valid descriptors.
        let (queue, descriptor) =
            unsafe { (MFCreateEventQueue()?, MFCreatePresentationDescriptor(Some(&[Some(stream.descriptor.clone())]))?) };
        *lock(&source.state) = SourceState { queue: Some(queue), descriptor: Some(descriptor), stream: Some(stream) };
        Ok(source)
    }

    /// The queue, the descriptor and the stream, or `MF_E_SHUTDOWN`.
    fn parts(&self) -> Result<(IMFEventQueue, IMFPresentationDescriptor, ComObject<Stream>)> {
        let state = lock(&self.state);
        match (&state.queue, &state.descriptor, &state.stream) {
            (Some(queue), Some(descriptor), Some(stream)) => Ok((queue.clone(), descriptor.clone(), stream.clone())),
            _ => Err(MF_E_SHUTDOWN.into()),
        }
    }

    fn queue(&self) -> Result<IMFEventQueue> {
        lock(&self.state).queue.clone().ok_or_else(|| MF_E_SHUTDOWN.into())
    }
}

delegate_attributes!(Source_Impl);

impl IMFMediaEventGenerator_Impl for Source_Impl {
    fn GetEvent(&self, flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS) -> Result<IMFMediaEvent> {
        // Outside the lock: this call may block until an event arrives.
        let queue = self.queue()?;
        // SAFETY: a valid queue.
        unsafe { queue.GetEvent(flags.0) }
    }

    fn BeginGetEvent(&self, callback: Ref<IMFAsyncCallback>, state: Ref<IUnknown>) -> Result<()> {
        // SAFETY: a valid queue; the caller's callback and state are passed on.
        unsafe { self.queue()?.BeginGetEvent(callback.as_ref(), state.as_ref()) }
    }

    fn EndGetEvent(&self, result: Ref<IMFAsyncResult>) -> Result<IMFMediaEvent> {
        // SAFETY: a valid queue; the caller's result is passed on.
        unsafe { self.queue()?.EndGetEvent(result.as_ref()) }
    }

    fn QueueEvent(&self, kind: u32, extended: *const GUID, status: HRESULT, value: *const PROPVARIANT) -> Result<()> {
        // SAFETY: a valid queue; the caller's pointers are passed on unchanged.
        unsafe { self.queue()?.QueueEventParamVar(kind, extended, status, value) }
    }
}

impl IMFMediaSource_Impl for Source_Impl {
    fn GetCharacteristics(&self) -> Result<u32> {
        Ok(MFMEDIASOURCE_IS_LIVE.0 as u32)
    }

    fn CreatePresentationDescriptor(&self) -> Result<IMFPresentationDescriptor> {
        let (_, descriptor, _) = self.parts()?;
        // SAFETY: a valid descriptor.
        unsafe { descriptor.Clone() }
    }

    fn Start(&self, descriptor: Ref<IMFPresentationDescriptor>, time_format: *const GUID, _start: *const PROPVARIANT) -> Result<()> {
        let requested = descriptor.ok()?;
        // SAFETY: COM passes either null or a readable GUID.
        if !time_format.is_null() && unsafe { time_format.read() } != GUID::zeroed() {
            return Err(E_INVALIDARG.into());
        }
        let (queue, ours, stream) = self.parts()?;
        let mut selected = BOOL::default();
        let mut stream_descriptor = None;
        // SAFETY: valid descriptors and out pointers; the stream object outlives the event.
        unsafe {
            requested.GetStreamDescriptorByIndex(STREAM_ID, &raw mut selected, &raw mut stream_descriptor)?;
            if selected.as_bool() {
                ours.SelectStream(STREAM_ID)?;
                let event = if stream.is_running() { MEUpdatedStream } else { MENewStream };
                queue.QueueEventParamUnk(event.0 as u32, &GUID::zeroed(), S_OK, &stream.to_interface::<IUnknown>())?;
                stream.start()?;
            } else {
                ours.DeselectStream(STREAM_ID)?;
                stream.stop()?;
            }
            queue.QueueEventParamVar(MESourceStarted.0 as u32, &GUID::zeroed(), S_OK, &time_value(now()))
        }
    }

    fn Stop(&self) -> Result<()> {
        let (queue, ours, stream) = self.parts()?;
        stream.stop()?;
        // SAFETY: valid descriptor and queue.
        unsafe {
            ours.DeselectStream(STREAM_ID)?;
            queue.QueueEventParamVar(MESourceStopped.0 as u32, &GUID::zeroed(), S_OK, &time_value(now()))
        }
    }

    fn Pause(&self) -> Result<()> {
        Err(MF_E_INVALID_STATE_TRANSITION.into())
    }

    fn Shutdown(&self) -> Result<()> {
        let state = std::mem::take(&mut *lock(&self.state));
        let queue = state.queue.ok_or(MF_E_SHUTDOWN)?;
        // SAFETY: a valid queue, shut down once.
        let _ = unsafe { queue.Shutdown() };
        if let Some(stream) = state.stream {
            stream.shutdown();
        }
        Ok(())
    }
}

impl IMFMediaSourceEx_Impl for Source_Impl {
    fn GetSourceAttributes(&self) -> Result<IMFAttributes> {
        Ok(self.attributes.clone())
    }

    fn GetStreamAttributes(&self, id: u32) -> Result<IMFAttributes> {
        if id != STREAM_ID {
            return Err(MF_E_INVALIDSTREAMNUMBER.into());
        }
        let (_, _, stream) = self.parts()?;
        Ok(stream.attributes.clone())
    }

    fn SetD3DManager(&self, _manager: Ref<IUnknown>) -> Result<()> {
        // Samples stay in system memory; the frame server uploads them where it needs to.
        Ok(())
    }
}

impl IMFGetService_Impl for Source_Impl {
    fn GetService(&self, _service: *const GUID, _iid: *const GUID, _object: *mut *mut c_void) -> Result<()> {
        Err(MF_E_UNSUPPORTED_SERVICE.into())
    }
}

impl IKsControl_Impl for Source_Impl {
    fn KsProperty(&self, _: *const KSIDENTIFIER, _: u32, _: *mut c_void, _: u32, _: *mut u32) -> Result<()> {
        no_ks_support()
    }
    fn KsMethod(&self, _: *const KSIDENTIFIER, _: u32, _: *mut c_void, _: u32, _: *mut u32) -> Result<()> {
        no_ks_support()
    }
    fn KsEvent(&self, _: *const KSIDENTIFIER, _: u32, _: *mut c_void, _: u32, _: *mut u32) -> Result<()> {
        no_ks_support()
    }
}

// ── Stream ──────────────────────────────────────────────────────────────────────────────────

#[implement(IMFMediaStream2, IKsControl, IMFAttributes)]
struct Stream {
    attributes: IMFAttributes,
    descriptor: IMFStreamDescriptor,
    media_type: IMFMediaType,
    header: Header,
    state: Mutex<StreamState>,
}

struct StreamState {
    /// The owning source (a reference cycle, broken by `shutdown`).
    source: Option<IMFMediaSource>,
    queue: Option<IMFEventQueue>,
    allocator: Option<IMFVideoSampleAllocatorEx>,
    running: MF_STREAM_STATE,
    frames: Option<Mapping>,
    /// When the frames file was last looked for, if it was missing.
    looked_for_frames: Option<Instant>,
}

impl Stream {
    fn new(source: IMFMediaSource, header: Header) -> Result<ComObject<Self>> {
        let attributes = new_attributes()?;
        // SAFETY: valid, freshly created Media Foundation objects throughout.
        let (media_type, descriptor, queue) = unsafe {
            attributes.SetGUID(&MF_DEVICESTREAM_STREAM_CATEGORY, &PINNAME_VIDEO_CAPTURE)?;
            attributes.SetUINT32(&MF_DEVICESTREAM_STREAM_ID, STREAM_ID)?;
            attributes.SetUINT32(&MF_DEVICESTREAM_FRAMESERVER_SHARED, 1)?;
            attributes.SetUINT32(&MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES, MFFrameSourceTypes_Color.0 as u32)?;

            let media_type = MFCreateMediaType()?;
            media_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            media_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
            media_type.SetUINT64(&MF_MT_FRAME_SIZE, packed(header.width, header.height))?;
            media_type.SetUINT64(&MF_MT_FRAME_RATE, packed(header.fps, 1))?;
            media_type.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, packed(1, 1))?;
            media_type.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
            media_type.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1)?;
            media_type.SetUINT32(&MF_MT_DEFAULT_STRIDE, header.width)?;
            let bits_per_second = header.frame_len() as u64 * u64::from(u8::BITS) * u64::from(header.fps);
            media_type.SetUINT32(&MF_MT_AVG_BITRATE, u32::try_from(bits_per_second).unwrap_or(u32::MAX))?;

            let descriptor = MFCreateStreamDescriptor(STREAM_ID, &[Some(media_type.clone())])?;
            descriptor.GetMediaTypeHandler()?.SetCurrentMediaType(&media_type)?;
            (media_type, descriptor, MFCreateEventQueue()?)
        };
        let state = StreamState {
            source: Some(source),
            queue: Some(queue),
            allocator: None,
            running: MF_STREAM_STATE_STOPPED,
            frames: None,
            looked_for_frames: None,
        };
        Ok(ComObject::new(Self { attributes, descriptor, media_type, header, state: Mutex::new(state) }))
    }

    fn is_running(&self) -> bool {
        lock(&self.state).running == MF_STREAM_STATE_RUNNING
    }

    /// Ready the sample pool (once) and report the stream started.
    fn start(&self) -> Result<()> {
        let mut state = lock(&self.state);
        let queue = state.queue.clone().ok_or(MF_E_SHUTDOWN)?;
        if state.allocator.is_none() {
            let mut allocator = std::ptr::null_mut();
            // SAFETY: a valid IID and out pointer; on success it holds an owned
            // IMFVideoSampleAllocatorEx reference, which `from_raw` takes over.
            let allocator = unsafe {
                MFCreateVideoSampleAllocatorEx(&IMFVideoSampleAllocatorEx::IID, &raw mut allocator)?;
                let allocator = IMFVideoSampleAllocatorEx::from_raw(allocator);
                allocator.InitializeSampleAllocator(SAMPLE_POOL, &self.media_type)?;
                allocator
            };
            state.allocator = Some(allocator);
        }
        state.running = MF_STREAM_STATE_RUNNING;
        drop(state);
        // SAFETY: a valid queue.
        unsafe { queue.QueueEventParamVar(MEStreamStarted.0 as u32, &GUID::zeroed(), S_OK, std::ptr::null()) }
    }

    /// Release the sample pool and report the stream stopped.
    fn stop(&self) -> Result<()> {
        let mut state = lock(&self.state);
        let queue = state.queue.clone().ok_or(MF_E_SHUTDOWN)?;
        if let Some(allocator) = state.allocator.take() {
            // SAFETY: a valid allocator, released once.
            let _ = unsafe { allocator.UninitializeSampleAllocator() };
        }
        let was_running = state.running == MF_STREAM_STATE_RUNNING;
        state.running = MF_STREAM_STATE_STOPPED;
        drop(state);
        if !was_running {
            return Ok(());
        }
        // SAFETY: a valid queue.
        unsafe { queue.QueueEventParamVar(MEStreamStopped.0 as u32, &GUID::zeroed(), S_OK, std::ptr::null()) }
    }

    /// Drop everything; later calls fail with `MF_E_SHUTDOWN`.
    fn shutdown(&self) {
        let mut state = lock(&self.state);
        if let Some(queue) = state.queue.take() {
            // SAFETY: a valid queue, shut down once.
            let _ = unsafe { queue.Shutdown() };
        }
        if let Some(allocator) = state.allocator.take() {
            // SAFETY: a valid allocator, released once.
            let _ = unsafe { allocator.UninitializeSampleAllocator() };
        }
        state.source = None;
        state.frames = None;
        state.running = MF_STREAM_STATE_STOPPED;
    }

    /// Fill `buffer` with the newest frame, or black.
    fn fill(&self, state: &mut StreamState, buffer: &IMFMediaBuffer) -> Result<()> {
        let header = self.header;
        if let Ok(buffer) = buffer.cast::<IMF2DBuffer2>() {
            let (mut scanline, mut pitch, mut start, mut len) = (std::ptr::null_mut(), 0i32, std::ptr::null_mut(), 0u32);
            // SAFETY: valid out pointers; on success the buffer stays locked (and the pointers
            // valid for `len` bytes from `start`) until `Unlock2D` below.
            unsafe { buffer.Lock2DSize(MF2DBuffer_LockFlags_Write, &raw mut scanline, &raw mut pitch, &raw mut start, &raw mut len)? };
            let offset = (scanline as usize).checked_sub(start as usize);
            if let (Some(offset), Ok(pitch)) = (offset, usize::try_from(pitch))
                && offset <= len as usize
            {
                // SAFETY: the locked buffer spans `len` bytes from `start`, writable, and nothing
                // else touches it until it is unlocked.
                let bytes = unsafe { std::slice::from_raw_parts_mut(start, len as usize) };
                let rows = &mut bytes[offset..];
                if !self.read_frame(state, rows, pitch) {
                    let _ = layout::fill_black(header, rows, pitch);
                }
            }
            // SAFETY: pairs the successful lock above.
            unsafe { buffer.Unlock2D()? };
            return Ok(());
        }
        let (mut data, mut capacity) = (std::ptr::null_mut(), 0u32);
        // SAFETY: valid out pointers; the buffer stays locked until `Unlock` below.
        unsafe { buffer.Lock(&raw mut data, Some(&raw mut capacity), None)? };
        // SAFETY: the locked buffer spans `capacity` writable bytes from `data`.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, capacity as usize) };
        let width = header.width as usize;
        if !self.read_frame(state, bytes, width) {
            let _ = layout::fill_black(header, bytes, width);
        }
        // SAFETY: pairs the successful lock above; the length is within the buffer.
        unsafe {
            buffer.Unlock()?;
            buffer.SetCurrentLength((header.frame_len() as u32).min(capacity))
        }
    }

    /// Copy the newest whole frame into `rows` (`pitch` bytes apart). `false` if there is none.
    fn read_frame(&self, state: &mut StreamState, rows: &mut [u8], pitch: usize) -> bool {
        if state.frames.is_none() && state.looked_for_frames.is_none_or(|at| at.elapsed() >= REOPEN_INTERVAL) {
            state.frames = open_frames();
            state.looked_for_frames = state.frames.is_none().then(Instant::now);
        }
        let Some(frames) = &state.frames else { return false };
        let Some(header) = Header::parse(&frames.header_bytes()) else { return false };
        if header.file_len() > frames.len() {
            // wessh grew the file for a larger size; map it again next time.
            state.frames = None;
            return false;
        }
        if header != self.header {
            return false;
        }
        for _ in 0..READ_ATTEMPTS {
            let latest = frames.latest().load(Ordering::Acquire);
            if latest == 0 {
                return false;
            }
            let Some(frame) = frames.bytes(layout::frame_range(header, latest)) else { return false };
            if !layout::copy_nv12(frame, header, rows, pitch) {
                return false;
            }
            fence(Ordering::Acquire);
            if !layout::is_torn(latest, frames.writing().load(Ordering::Relaxed)) {
                return true;
            }
        }
        false
    }
}

delegate_attributes!(Stream_Impl);

impl Stream_Impl {
    fn queue(&self) -> Result<IMFEventQueue> {
        lock(&self.state).queue.clone().ok_or_else(|| MF_E_SHUTDOWN.into())
    }
}

impl IMFMediaEventGenerator_Impl for Stream_Impl {
    fn GetEvent(&self, flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS) -> Result<IMFMediaEvent> {
        // Outside the lock: this call may block until an event arrives.
        let queue = self.queue()?;
        // SAFETY: a valid queue.
        unsafe { queue.GetEvent(flags.0) }
    }

    fn BeginGetEvent(&self, callback: Ref<IMFAsyncCallback>, state: Ref<IUnknown>) -> Result<()> {
        // SAFETY: a valid queue; the caller's callback and state are passed on.
        unsafe { self.queue()?.BeginGetEvent(callback.as_ref(), state.as_ref()) }
    }

    fn EndGetEvent(&self, result: Ref<IMFAsyncResult>) -> Result<IMFMediaEvent> {
        // SAFETY: a valid queue; the caller's result is passed on.
        unsafe { self.queue()?.EndGetEvent(result.as_ref()) }
    }

    fn QueueEvent(&self, kind: u32, extended: *const GUID, status: HRESULT, value: *const PROPVARIANT) -> Result<()> {
        // SAFETY: a valid queue; the caller's pointers are passed on unchanged.
        unsafe { self.queue()?.QueueEventParamVar(kind, extended, status, value) }
    }
}

impl IMFMediaStream_Impl for Stream_Impl {
    fn GetMediaSource(&self) -> Result<IMFMediaSource> {
        lock(&self.state).source.clone().ok_or_else(|| MF_E_SHUTDOWN.into())
    }

    fn GetStreamDescriptor(&self) -> Result<IMFStreamDescriptor> {
        Ok(self.descriptor.clone())
    }

    fn RequestSample(&self, token: Ref<IUnknown>) -> Result<()> {
        let mut state = lock(&self.state);
        let (Some(queue), Some(allocator)) = (state.queue.clone(), state.allocator.clone()) else {
            return Err(MF_E_SHUTDOWN.into());
        };
        // SAFETY: valid sample, buffer and queue objects; the sample is handed to the queue,
        // which keeps its own reference.
        unsafe {
            let sample = allocator.AllocateSample()?;
            self.fill(&mut state, &sample.GetBufferByIndex(0)?)?;
            drop(state);
            sample.SetSampleTime(now())?;
            sample.SetSampleDuration(TICKS_PER_SECOND / i64::from(self.header.fps))?;
            if let Some(token) = token.as_ref() {
                sample.SetUnknown(&MFSampleExtension_Token, token)?;
            }
            queue.QueueEventParamUnk(MEMediaSample.0 as u32, &GUID::zeroed(), S_OK, &sample)
        }
    }
}

impl IMFMediaStream2_Impl for Stream_Impl {
    fn SetStreamState(&self, value: MF_STREAM_STATE) -> Result<()> {
        match value {
            MF_STREAM_STATE_RUNNING => self.start(),
            MF_STREAM_STATE_STOPPED => self.stop(),
            _ => Err(MF_E_INVALID_STATE_TRANSITION.into()),
        }
    }

    fn GetStreamState(&self) -> Result<MF_STREAM_STATE> {
        Ok(lock(&self.state).running)
    }
}

impl IKsControl_Impl for Stream_Impl {
    fn KsProperty(&self, _: *const KSIDENTIFIER, _: u32, _: *mut c_void, _: u32, _: *mut u32) -> Result<()> {
        no_ks_support()
    }
    fn KsMethod(&self, _: *const KSIDENTIFIER, _: u32, _: *mut c_void, _: u32, _: *mut u32) -> Result<()> {
        no_ks_support()
    }
    fn KsEvent(&self, _: *const KSIDENTIFIER, _: u32, _: *mut c_void, _: u32, _: *mut u32) -> Result<()> {
        no_ks_support()
    }
}
