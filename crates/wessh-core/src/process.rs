//! Commands used for background downloads, hardware probes and media processing.

use std::ffi::OsStr;
use std::process::Command;

/// Keep piped/inherited I/O, without opening a console window on Windows.
pub fn background_command(program: impl AsRef<OsStr>) -> Command {
    let command = Command::new(program);
    #[cfg(windows)]
    let command = {
        use std::os::windows::process::CommandExt;
        // Win32 CREATE_NO_WINDOW: console helpers must not flash windows over the GUI.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut command = command;
        command.creation_flags(CREATE_NO_WINDOW);
        command
    };
    command
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn helper_has_no_console_and_keeps_output_and_exit_status() {
        let script = r#"
Add-Type 'using System; using System.Runtime.InteropServices; public class ConsoleProbe { [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); }'
[Console]::Out.WriteLine([ConsoleProbe]::GetConsoleWindow().ToInt64())
[Console]::Error.WriteLine('background stderr')
exit 7
"#;
        let output = background_command("powershell.exe").args(["-NoProfile", "-NonInteractive", "-Command", script]).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "0");
        assert!(String::from_utf8_lossy(&output.stderr).contains("background stderr"));
        assert_eq!(output.status.code(), Some(7));
    }
}
