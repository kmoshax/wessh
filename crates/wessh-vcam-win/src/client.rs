//! What runs in wessh: register the virtual camera, write frames, install the media source.

use crate::layout::{self, Header, PROTOCOL};
use crate::shared::Mapping;
use crate::{CLSID_STRING, DLL_NAME, Error, FRIENDLY_NAME, PROTOCOL_VALUE, Result, Status};
use std::ffi::{OsStr, c_void};
use std::fs::{self, File, OpenOptions};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{Ordering, fence};
use windows::Win32::Foundation::{CloseHandle, ERROR_CANCELLED, ERROR_FILE_NOT_FOUND, HANDLE};
use windows::Win32::Media::MediaFoundation::{
    IMFVirtualCamera, MF_VERSION, MFSTARTUP_LITE, MFStartup, MFVirtualCameraAccess, MFVirtualCameraAccess_CurrentUser,
    MFVirtualCameraLifetime, MFVirtualCameraLifetime_Session, MFVirtualCameraType, MFVirtualCameraType_SoftwareCameraSource,
};
use windows::Win32::Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation};
use windows::Win32::Storage::FileSystem::{
    FILE_ATTRIBUTE_TEMPORARY, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, MOVEFILE_DELAY_UNTIL_REBOOT, MoveFileExW,
};
use windows::Win32::System::Com::{COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoIncrementMTAUsage, CoInitializeEx, CoUninitialize};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW};
use windows::Win32::System::Registry::{
    HKEY, HKEY_LOCAL_MACHINE, KEY_WOW64_64KEY, KEY_WRITE, REG_DWORD, REG_OPTION_NON_VOLATILE, REG_SZ, REG_VALUE_TYPE, RRF_RT_REG_DWORD,
    RRF_RT_REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegGetValueW, RegSetValueExW,
};
use windows::Win32::System::Threading::{GetCurrentProcess, GetExitCodeProcess, INFINITE, OpenProcessToken, WaitForSingleObject};
use windows::Win32::UI::Shell::{SEE_MASK_NOASYNC, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW, ShellExecuteExW};
use windows::Win32::UI::WindowsAndMessaging::SW_HIDE;
use windows_core::{GUID, HRESULT, Interface, PCSTR, PCWSTR, s, w};

/// The library that exports the virtual camera API (Windows 11 and later only).
const SENSOR_GROUP_DLL: PCWSTR = w!("mfsensorgroup.dll");

/// `MFCreateVirtualCamera`, looked up at run time: binding it at load time would stop wessh
/// from starting at all on Windows 10.
type CreateVirtualCamera = unsafe extern "system" fn(
    MFVirtualCameraType,
    MFVirtualCameraLifetime,
    MFVirtualCameraAccess,
    PCWSTR,
    PCWSTR,
    *const GUID,
    u32,
    *mut *mut c_void,
) -> HRESULT;
/// `MFIsVirtualCameraTypeSupported`, also Windows 11 only.
type IsVirtualCameraTypeSupported = unsafe extern "system" fn(MFVirtualCameraType, *mut windows_core::BOOL) -> HRESULT;

/// A function exported by `mfsensorgroup.dll`, if this Windows has it.
fn sensor_group_export(name: PCSTR) -> Option<unsafe extern "system" fn() -> isize> {
    // Kept as an address: the module stays loaded for the life of the process.
    static MODULE: OnceLock<Option<usize>> = OnceLock::new();
    let module = (*MODULE.get_or_init(|| {
        // SAFETY: loading a system DLL by name from System32 only; the handle is never freed.
        unsafe { LoadLibraryExW(SENSOR_GROUP_DLL, None, LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok().map(|m| m.0 as usize)
    }))?;
    // SAFETY: `module` is a loaded module handle and `name` a NUL-terminated export name.
    unsafe { GetProcAddress(windows::Win32::Foundation::HMODULE(module as *mut c_void), name) }
}

/// Whether this Windows can create software virtual cameras (Windows 11).
fn os_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        let (Some(_), Some(is_supported)) =
            (sensor_group_export(s!("MFCreateVirtualCamera")), sensor_group_export(s!("MFIsVirtualCameraTypeSupported")))
        else {
            return false;
        };
        // SAFETY: the export has this signature (mfvirtualcamera.h).
        let is_supported: IsVirtualCameraTypeSupported = unsafe { std::mem::transmute(is_supported) };
        let mut supported = windows_core::BOOL(0);
        // SAFETY: `supported` is a valid out pointer for the call.
        let result = unsafe { is_supported(MFVirtualCameraType_SoftwareCameraSource, &raw mut supported) };
        result.is_ok() && supported.as_bool()
    })
}

/// Whether the virtual camera can be used now.
#[must_use]
pub fn status() -> Status {
    if !os_supported() {
        return Status::Unsupported;
    }
    match installed_protocol() {
        None => Status::NotInstalled,
        Some(PROTOCOL) => Status::Ready,
        Some(_) => Status::Outdated,
    }
}

/// `Software\Classes\CLSID\{…}` under `HKLM`.
fn class_key() -> String {
    format!(r"Software\Classes\CLSID\{CLSID_STRING}")
}

fn inproc_key() -> String {
    format!(r"{}\InprocServer32", class_key())
}

/// The protocol the registered source speaks, if it is registered and its DLL exists.
fn installed_protocol() -> Option<u32> {
    let dll = read_string(&inproc_key())?;
    if !Path::new(&dll).is_file() {
        return None;
    }
    let mut value = 0u32;
    let mut size = size_of::<u32>() as u32;
    let (key, name) = (wide(OsStr::new(&class_key())), wide(OsStr::new(PROTOCOL_VALUE)));
    // SAFETY: NUL-terminated key and value names; `value` and `size` are valid for a DWORD.
    let result = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            PCWSTR(key.as_ptr()),
            PCWSTR(name.as_ptr()),
            RRF_RT_REG_DWORD,
            None,
            Some((&raw mut value).cast()),
            Some(&raw mut size),
        )
    };
    result.is_ok().then_some(value)
}

/// The default string value of `HKLM\<key>`.
fn read_string(key: &str) -> Option<String> {
    /// Longest path we read back (Windows' extended path limit, in UTF-16 units).
    const MAX_VALUE: usize = 32_768;
    let key = wide(OsStr::new(key));
    let mut buffer = vec![0u16; MAX_VALUE];
    let mut size = (buffer.len() * 2) as u32;
    // SAFETY: NUL-terminated key name; `buffer` holds `size` bytes.
    let result = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            PCWSTR(key.as_ptr()),
            PCWSTR::null(),
            RRF_RT_REG_SZ,
            None,
            Some(buffer.as_mut_ptr().cast()),
            Some(&raw mut size),
        )
    };
    if result.is_err() {
        return None;
    }
    let units = (size as usize / 2).min(buffer.len());
    let text = String::from_utf16_lossy(&buffer[..units]);
    Some(text.trim_end_matches('\0').to_owned())
}

fn wide(text: &OsStr) -> Vec<u16> {
    text.encode_wide().chain(std::iter::once(0)).collect()
}

/// Start COM (multithreaded, for every thread of the process) and Media Foundation, once.
fn start_media_foundation() -> Result<()> {
    static STARTED: OnceLock<Result<(), windows_core::Error>> = OnceLock::new();
    STARTED
        .get_or_init(|| {
            // SAFETY: plain initialisation calls; the MTA usage and MF are kept for the life of
            // the process (never shut down), so no pairing calls are needed.
            unsafe {
                CoIncrementMTAUsage()?;
                MFStartup(MF_VERSION, MFSTARTUP_LITE)
            }
        })
        .clone()
        .map_err(Error::from)
}

/// The registered virtual camera and the frames file feeding it. The camera disappears from
/// other apps when this is dropped (or the process exits).
pub struct VirtualCamera {
    camera: IMFVirtualCamera,
    frames: Mapping,
    header: Header,
    /// The frame [`VirtualCamera::frame`] hands out next.
    next: u64,
}

// SAFETY: the camera object is created in the process-wide multithreaded apartment
// (`CoIncrementMTAUsage`), so it may be used and released from any thread.
unsafe impl Send for VirtualCamera {}

impl std::fmt::Debug for VirtualCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualCamera").field("header", &self.header).field("next", &self.next).finish_non_exhaustive()
    }
}

impl VirtualCamera {
    /// Register the camera for `width`×`height` NV12 frames at `fps`. Needs Windows 11 and the
    /// installed media source; no administrator rights.
    pub fn start(width: u32, height: u32, fps: u32) -> Result<Self> {
        let header = Header { width, height, fps };
        if !header.is_valid() {
            return Err(Error::Format { width, height, fps });
        }
        match status() {
            Status::Unsupported => return Err(Error::Unsupported),
            Status::NotInstalled | Status::Outdated => return Err(Error::NotInstalled),
            Status::Ready => {}
        }
        start_media_foundation()?;
        let frames = open_frames(header)?;
        let create = sensor_group_export(s!("MFCreateVirtualCamera")).ok_or(Error::Unsupported)?;
        // SAFETY: the export has this signature (mfvirtualcamera.h).
        let create: CreateVirtualCamera = unsafe { std::mem::transmute(create) };
        let (name, source) = (wide(OsStr::new(FRIENDLY_NAME)), wide(OsStr::new(CLSID_STRING)));
        let mut camera = std::ptr::null_mut();
        // SAFETY: NUL-terminated strings that outlive the call, no categories, and a valid out
        // pointer that receives an owned reference on success.
        let camera = unsafe {
            create(
                MFVirtualCameraType_SoftwareCameraSource,
                MFVirtualCameraLifetime_Session,
                MFVirtualCameraAccess_CurrentUser,
                PCWSTR(name.as_ptr()),
                PCWSTR(source.as_ptr()),
                std::ptr::null(),
                0,
                &raw mut camera,
            )
            .ok()?;
            IMFVirtualCamera::from_raw(camera)
        };
        // SAFETY: a valid camera object; no callback, so the call completes before returning.
        unsafe { camera.Start(None)? };
        Ok(Self { camera, frames, header, next: 1 })
    }

    /// The NV12 buffer for the next frame (`width * height * 3 / 2` bytes): fill it, then
    /// [`VirtualCamera::publish`]. It is shared memory, so writing here is the only copy.
    pub fn frame(&mut self) -> &mut [u8] {
        self.frames.writing().store(self.next, Ordering::Relaxed);
        // Readers that see this frame's bytes also see `writing`, so they can tell a torn copy.
        fence(Ordering::Release);
        let range = layout::frame_range(self.header, self.next);
        self.frames.bytes_mut(range).unwrap_or_default()
    }

    /// Make the frame written into [`VirtualCamera::frame`] the one apps receive.
    pub fn publish(&mut self) {
        self.frames.latest().store(self.next, Ordering::Release);
        self.next += 1;
    }
}

impl Drop for VirtualCamera {
    fn drop(&mut self) {
        // SAFETY: a valid camera object. `Remove` unregisters it (not `Shutdown`, which would
        // shut the media source down twice).
        let _ = unsafe { self.camera.Remove() };
    }
}

/// Create or reuse the frames file, sized for `header`, and reset it to "no frame yet".
fn open_frames(header: Header) -> Result<Mapping> {
    let path = crate::frames_path();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    // Readers may still map the file from an earlier session, so it is reused, never truncated
    // (a mapped file cannot shrink); growing it is allowed.
    let file: File = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .share_mode((FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE).0)
        .attributes(FILE_ATTRIBUTE_TEMPORARY.0)
        .open(&path)?;
    let len = header.file_len();
    if file.metadata()?.len() < len as u64 {
        file.set_len(len as u64)?;
    }
    let mut frames = Mapping::new(&file, len, true)?;
    // No frame first, so a reader never pairs the new size with old pixels.
    frames.latest().store(0, Ordering::Release);
    frames.writing().store(0, Ordering::Release);
    frames.write_header(header);
    Ok(frames)
}

/// Install the media source from `dll`: copy it to [`crate::install_dir`] and register it.
/// Needs administrator rights ([`run_elevated`]). Returns where it was installed.
pub fn install(dll: &Path) -> Result<PathBuf> {
    if !os_supported() {
        return Err(Error::Unsupported);
    }
    // Read and written rather than copied, so the installed file has no "downloaded from the
    // internet" mark, which the frame server refuses to load.
    let bytes = fs::read(dll)?;
    let dir = crate::install_dir();
    fs::create_dir_all(&dir)?;
    remove_leftovers(&dir);
    let target = dir.join(DLL_NAME);
    let fresh = dir.join(format!("{DLL_NAME}.new"));
    fs::write(&fresh, bytes)?;
    set_aside(&target)?;
    fs::rename(&fresh, &target)?;
    register(&target)?;
    Ok(target)
}

/// Unregister the media source and delete it (at the next restart if it is in use).
pub fn uninstall() -> Result<()> {
    let key = wide(OsStr::new(&class_key()));
    // SAFETY: NUL-terminated key name under HKLM.
    let result = unsafe { RegDeleteTreeW(HKEY_LOCAL_MACHINE, PCWSTR(key.as_ptr())) };
    if result != ERROR_FILE_NOT_FOUND {
        result.ok()?;
    }
    let dir = crate::install_dir();
    set_aside(&dir.join(DLL_NAME))?;
    remove_leftovers(&dir);
    let _ = fs::remove_dir(&dir);
    Ok(())
}

/// Suffix of DLLs set aside because a process still had them loaded.
const OLD_SUFFIX: &str = ".old-";

/// Delete `path`, or if it is loaded somewhere, move it aside and delete it at the next restart.
fn set_aside(path: &Path) -> Result<()> {
    if !path.exists() || fs::remove_file(path).is_ok() {
        return Ok(());
    }
    let old = path.with_file_name(format!("{DLL_NAME}{OLD_SUFFIX}{}", std::process::id()));
    fs::rename(path, &old)?;
    let old = wide(old.as_os_str());
    // SAFETY: NUL-terminated path; a null new name schedules deletion at restart.
    let _ = unsafe { MoveFileExW(PCWSTR(old.as_ptr()), PCWSTR::null(), MOVEFILE_DELAY_UNTIL_REBOOT) };
    Ok(())
}

/// Delete DLLs earlier installs set aside, where nothing loads them any more.
fn remove_leftovers(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(&format!("{DLL_NAME}{OLD_SUFFIX}")) {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// Register the COM class under HKLM (the frame server ignores per-user registrations).
fn register(dll: &Path) -> Result<()> {
    let class = Key::create(&class_key())?;
    class.set_string(None, OsStr::new("wessh virtual camera"))?;
    let inproc = Key::create(&inproc_key())?;
    inproc.set_string(None, dll.as_os_str())?;
    inproc.set_string(Some("ThreadingModel"), OsStr::new("Both"))?;
    // Last, so a half-finished install reads as not installed.
    class.set(Some(PROTOCOL_VALUE), REG_DWORD, &PROTOCOL.to_le_bytes())
}

/// An open registry key, closed on drop.
struct Key(HKEY);

impl Key {
    fn create(path: &str) -> Result<Self> {
        let path = wide(OsStr::new(path));
        let mut key = HKEY::default();
        // SAFETY: NUL-terminated subkey name and a valid out pointer for the key.
        unsafe {
            RegCreateKeyExW(
                HKEY_LOCAL_MACHINE,
                PCWSTR(path.as_ptr()),
                None,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_WRITE | KEY_WOW64_64KEY,
                None,
                &raw mut key,
                None,
            )
            .ok()?;
        }
        Ok(Self(key))
    }

    fn set_string(&self, name: Option<&str>, value: &OsStr) -> Result<()> {
        let bytes: Vec<u8> = wide(value).iter().flat_map(|unit| unit.to_le_bytes()).collect();
        self.set(name, REG_SZ, &bytes)
    }

    fn set(&self, name: Option<&str>, kind: REG_VALUE_TYPE, data: &[u8]) -> Result<()> {
        let name = name.map(|n| wide(OsStr::new(n)));
        let name = name.as_ref().map_or(PCWSTR::null(), |n| PCWSTR(n.as_ptr()));
        // SAFETY: an open key, a NUL-terminated (or null, for the default) value name, and data
        // of the declared type.
        unsafe { RegSetValueExW(self.0, name, None, kind, Some(data)).ok()? };
        Ok(())
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: the key was opened by `Key::create` and is closed once.
        let _ = unsafe { RegCloseKey(self.0) };
    }
}

/// Whether this process runs with administrator rights.
#[must_use]
pub fn is_elevated() -> bool {
    let mut token = HANDLE::default();
    // SAFETY: the pseudo handle of this process and a valid out pointer.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw mut token) }.is_err() {
        return false;
    }
    let mut elevation = TOKEN_ELEVATION::default();
    let mut returned = 0u32;
    // SAFETY: `elevation` is the buffer TokenElevation expects, of the size passed.
    let queried = unsafe {
        GetTokenInformation(
            token,
            TokenElevation,
            Some((&raw mut elevation).cast()),
            size_of::<TOKEN_ELEVATION>() as u32,
            &raw mut returned,
        )
    };
    // SAFETY: `token` was opened above and is closed once.
    let _ = unsafe { CloseHandle(token) };
    queried.is_ok() && elevation.TokenIsElevated != 0
}

/// Run `program` with `arguments` as administrator (the UAC prompt), hidden, and wait for it.
/// Returns its exit code; [`Error::Cancelled`] if the prompt was declined.
pub fn run_elevated(program: &Path, arguments: &str) -> Result<u32> {
    // The shell wants COM on the calling thread; leave it as it was afterwards.
    // SAFETY: plain initialisation of this thread, undone below only if it succeeded.
    let initialised = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) }.is_ok();
    let result = shell_execute_elevated(program, arguments);
    if initialised {
        // SAFETY: pairs the successful `CoInitializeEx` above on this thread.
        unsafe { CoUninitialize() };
    }
    result
}

fn shell_execute_elevated(program: &Path, arguments: &str) -> Result<u32> {
    let (file, parameters) = (wide(program.as_os_str()), wide(OsStr::new(arguments)));
    let mut info = SHELLEXECUTEINFOW {
        cbSize: size_of::<SHELLEXECUTEINFOW>() as u32,
        fMask: SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC,
        lpVerb: w!("runas"),
        lpFile: PCWSTR(file.as_ptr()),
        lpParameters: PCWSTR(parameters.as_ptr()),
        nShow: SW_HIDE.0,
        ..Default::default()
    };
    // SAFETY: `info` is fully initialised and its strings outlive the call.
    if let Err(e) = unsafe { ShellExecuteExW(&raw mut info) } {
        return Err(if e.code() == ERROR_CANCELLED.to_hresult() { Error::Cancelled } else { e.into() });
    }
    let mut code = 0u32;
    // SAFETY: SEE_MASK_NOCLOSEPROCESS gives us the process handle, which we wait on, query and
    // close once.
    unsafe {
        WaitForSingleObject(info.hProcess, INFINITE);
        let exit = GetExitCodeProcess(info.hProcess, &raw mut code);
        let _ = CloseHandle(info.hProcess);
        exit?;
    }
    Ok(code)
}
