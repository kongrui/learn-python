@echo on
call C:\ProgramData\Anaconda3\Scripts\activate.bat
pushd %~dp0
set script_dir=%CD%
popd
C:\ProgramData\Anaconda3\python.exe %script_dir%\words.py
pause