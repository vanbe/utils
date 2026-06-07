@echo off
REM Construit capture.exe sous Windows (binaire NON versionne).
REM Essaie MinGW g++ puis MSVC cl.
setlocal
set HERE=%~dp0
set BIN=%HERE%..\bin
if not exist "%BIN%" mkdir "%BIN%"

where g++ >nul 2>nul
if %ERRORLEVEL%==0 (
  g++ -std=c++17 -O2 -static -static-libgcc -static-libstdc++ "%HERE%capture.cpp" -o "%BIN%\capture.exe" -lole32 -loleaut32 -luuid -lwinmm
  if %ERRORLEVEL%==0 ( echo OK -^> %BIN%\capture.exe & exit /b 0 )
  echo Echec g++ & exit /b 1
)

where cl >nul 2>nul
if %ERRORLEVEL%==0 (
  cl /nologo /std:c++17 /O2 /EHsc "%HERE%capture.cpp" /Fe:"%BIN%\capture.exe" ole32.lib oleaut32.lib
  if %ERRORLEVEL%==0 ( echo OK -^> %BIN%\capture.exe & exit /b 0 )
  echo Echec cl & exit /b 1
)

echo Aucun compilateur trouve (MinGW g++ ou MSVC cl). & exit /b 1
