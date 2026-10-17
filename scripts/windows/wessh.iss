; Inno Setup script for the Windows installer: a per-user install (no administrator rights) with
; Start menu and optional desktop shortcuts, the optional virtual camera (Windows 11; one UAC
; prompt, from `wessh camera install`), and an uninstaller that removes the camera and offers to
; remove the downloaded runtime and models too.
;
;   iscc /DAppVersion=0.3.0 /DSourceDir=dist\stage /DOutputDir=dist scripts\windows\wessh.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\stage"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{6E7F2B54-3C1A-4D8E-9B0F-5A1C2D3E4F60}
AppName=wessh
AppVersion={#AppVersion}
AppPublisher=kmoshax
AppPublisherURL=https://github.com/kmoshax/wessh
AppSupportURL=https://github.com/kmoshax/wessh/issues
DefaultDirName={localappdata}\Programs\wessh
DefaultGroupName=wessh
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=wessh-windows-x64-setup
SetupIconFile=..\..\crates\wessh-app\assets\icon\wessh.ico
UninstallDisplayIcon={app}\wessh.exe
LicenseFile=..\..\LICENSE
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "virtualcamera"; Description: "Add the wessh camera, so other apps can use wessh as a webcam (asks for administrator permission)"; Check: IsWindows11

[Files]
Source: "{#SourceDir}\wessh.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\wessh-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\wessh-vcam.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\wessh"; Filename: "{app}\wessh.exe"
Name: "{autodesktop}\wessh"; Filename: "{app}\wessh.exe"; Tasks: desktopicon

[Run]
; Installs or updates the camera only when needed, so silent updates don't prompt every time.
Filename: "{app}\wessh-cli.exe"; Parameters: "camera install"; StatusMsg: "Adding the wessh camera..."; Flags: runhidden; Tasks: virtualcamera; Check: CameraNeedsInstall
Filename: "{app}\wessh.exe"; Description: "{cm:LaunchProgram,wessh}"; Flags: nowait postinstall skipifsilent
; The in-app updater runs this installer silently with /relaunch=yes: reopen the app when done.
Filename: "{app}\wessh.exe"; Flags: nowait; Check: Relaunch

[UninstallRun]
Filename: "{app}\wessh-cli.exe"; Parameters: "camera uninstall"; Flags: runhidden; RunOnceId: "RemoveCamera"; Check: CameraInstalled

[Code]
// `wessh camera status` exit codes (crates/wessh-app/src/commands/camera.rs).
const
  CameraReady = 0;
  CameraNotInstalled = 3;
  CameraOutdated = 4;
  // Windows 11's first build, where MFCreateVirtualCamera appeared.
  Windows11Build = 22000;

function IsWindows11: Boolean;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  Result := Version.Build >= Windows11Build;
end;

function CameraStatus: Integer;
var
  Code: Integer;
begin
  if Exec(ExpandConstant('{app}\wessh-cli.exe'), 'camera status', '', SW_HIDE, ewWaitUntilTerminated, Code) then
    Result := Code
  else
    Result := -1;
end;

function CameraNeedsInstall: Boolean;
var
  Status: Integer;
begin
  Status := CameraStatus;
  Result := (Status = CameraNotInstalled) or (Status = CameraOutdated);
end;

function CameraInstalled: Boolean;
var
  Status: Integer;
begin
  Status := CameraStatus;
  Result := (Status = CameraReady) or (Status = CameraOutdated);
end;

function Relaunch: Boolean;
begin
  Result := ExpandConstant('{param:relaunch|no}') = 'yes';
end;

// The runtime, CUDA libraries and models live in %LOCALAPPDATA%\wessh and can take several GB.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\wessh');
    if DirExists(DataDir) and (not UninstallSilent) and (MsgBox('Also delete the downloaded face models and runtime (' + DataDir + ')?', mbConfirmation, MB_YESNO) = IDYES) then
      DelTree(DataDir, True, True, True);
  end;
end;
