; Inno Setup script for the Windows installer: a per-user install (no administrator rights) with
; Start menu and optional desktop shortcuts, and an uninstaller that offers to remove the
; downloaded runtime and models too.
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

[Files]
Source: "{#SourceDir}\wessh.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\wessh-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\wessh"; Filename: "{app}\wessh.exe"
Name: "{autodesktop}\wessh"; Filename: "{app}\wessh.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\wessh.exe"; Description: "{cm:LaunchProgram,wessh}"; Flags: nowait postinstall skipifsilent
; The in-app updater runs this installer silently with /relaunch=yes: reopen the app when done.
Filename: "{app}\wessh.exe"; Flags: nowait; Check: Relaunch

[Code]
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
