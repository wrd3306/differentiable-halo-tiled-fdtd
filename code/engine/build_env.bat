@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "KITS=C:\Program Files (x86)\Windows Kits\10"
set "SDK=10.0.26100.0"
set "INCLUDE=%INCLUDE%;%KITS%\Include\%SDK%\shared;%KITS%\Include\%SDK%\ucrt;%KITS%\Include\%SDK%\um"
set "LIB=%LIB%;%KITS%\Lib\%SDK%\ucrt\x64;%KITS%\Lib\%SDK%\um\x64"
set "PY=C:\Users\wrd\AppData\Local\Programs\Python\Python310"
set "PATH=%PY%;%PY%\Scripts;%PATH%"
set TORCH_DONT_CHECK_COMPILER_ABI=1
python -u %1