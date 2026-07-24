@echo off
setlocal

REM --------------------------------------------------------------------------
REM Launch the deepinv DRUNet / Set5 evaluation.
REM All paths come from the environment (so run_set5_deepinv.py stays generic).
REM Defaults below are relative to this script; override any of them by creating
REM a local env.bat (git-ignored -- see .gitignore) next to this file.
REM --------------------------------------------------------------------------
if exist "%~dp0env.bat" call "%~dp0env.bat"

if not defined PROJECT_DIR     set "PROJECT_DIR=%~dp0"
if not defined KAIR_ROOT      set "KAIR_ROOT=%PROJECT_DIR%..\KAIR"
if not defined SHIM_ROOT      set "SHIM_ROOT=%PROJECT_DIR%..\deepinv_shim"
if not defined DRUNET_WEIGHTS set "DRUNET_WEIGHTS=%KAIR_ROOT%\model_zoo\drunet_color.pth"
if not defined TESTSET_DIR    set "TESTSET_DIR=%KAIR_ROOT%\testsets\set5"
if not defined PYTHON         set "PYTHON=%KAIR_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] python not found: %PYTHON%
    exit /b 1
)

"%PYTHON%" "%PROJECT_DIR%run_set5_deepinv.py"
endlocal
