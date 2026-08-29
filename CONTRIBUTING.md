# Contributing

Bug reports and focused pull requests are welcome.

## Before opening a pull request

1. Keep changes compatible with Python 3.10 and newer.
2. Run the unit tests:

   ```powershell
   python -m unittest discover -s tests -v
   ```

3. Confirm all scripts still compile:

   ```powershell
   python -m py_compile whisper_trim.py whisper_trim_all.py whisper_trim_all_dependant.py
   ```

4. Update the README or script guide when behavior or command-line options change.

Please do not commit media files, generated outputs, model weights, virtual environments, or credentials.
