@echo off
setlocal EnableExtensions

rem Process media in the parent folder of this script's directory.
call "%~dp0trim_parent_folder.bat" %*
set "TRIM_EXIT_CODE=%errorlevel%"

echo.
pause
exit /b %TRIM_EXIT_CODE%
