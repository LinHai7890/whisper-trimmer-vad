@echo off
setlocal

call "%~dp0trim_parent_folder.bat" --model medium --padding 0.35 --merge-gap 0.7
exit /b %errorlevel%
