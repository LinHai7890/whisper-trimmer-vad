@echo off
python "%~dp0whisper_trim.py" "%~dp0.." --language en --model base --merge-gap 0.1 --padding 0.1
pause
