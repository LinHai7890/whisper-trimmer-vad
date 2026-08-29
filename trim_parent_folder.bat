@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PY_SCRIPT=%SCRIPT_DIR%whisper_trim.py"

for %%I in ("%SCRIPT_DIR%..") do set "SOURCE_DIR=%%~fI"
set "OUTPUT_DIR=%SOURCE_DIR%\trimmed"
set "TRIM_OPTIONS=%*"

if not exist "%PY_SCRIPT%" (
    echo Could not find whisper_trim.py beside this batch file:
    echo %PY_SCRIPT%
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Source folder: %SOURCE_DIR%
echo Output folder: %OUTPUT_DIR%
echo.

set /a FOUND=0
set /a DONE=0
set /a SKIPPED=0
set /a FAILED=0

for %%E in (.mp4 .mov .mkv .avi .webm .m4v .mp3 .wav .m4a .aac .flac .ogg .opus .wma) do (
    for %%F in ("%SOURCE_DIR%\*%%E") do (
        if exist "%%~fF" call :trim_one "%%~fF"
    )
)

echo.
echo Found:   %FOUND%
echo Done:    %DONE%
echo Skipped: %SKIPPED%
echo Failed:  %FAILED%

if %FAILED% NEQ 0 exit /b 1
exit /b 0

:trim_one
set "INPUT_FILE=%~1"
set "OUTPUT_FILE=%OUTPUT_DIR%\%~n1_trimmed%~x1"
set /a FOUND+=1

if exist "%OUTPUT_FILE%" (
    echo Skipping existing output: %~nx1
    set /a SKIPPED+=1
    exit /b 0
)

echo Trimming: %~nx1
if defined TRIM_OPTIONS (
    python "%PY_SCRIPT%" "%INPUT_FILE%" "%OUTPUT_FILE%" %TRIM_OPTIONS%
) else (
    python "%PY_SCRIPT%" "%INPUT_FILE%" "%OUTPUT_FILE%"
)

if errorlevel 1 (
    echo Failed: %~nx1
    set /a FAILED+=1
) else (
    set /a DONE+=1
)
exit /b 0
