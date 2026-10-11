//! Windows: put the app icon and version details into the executable, so Explorer, the taskbar
//! and the installer show them.

fn main() {
    println!("cargo:rerun-if-changed=assets/icon/wessh.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut resource = winresource::WindowsResource::new();
    resource.set_icon("assets/icon/wessh.ico").set("ProductName", "wessh").set("FileDescription", "wessh — real-time face swap");
    resource.compile().expect("compiling the Windows resources (needs the MSVC or MinGW resource compiler)");
}
