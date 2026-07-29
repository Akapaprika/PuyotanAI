@echo off
setlocal

REM =====================================
REM Build script for PuyotanNative
REM Usage:
REM   build.bat        -> Release build
REM   build.bat -d     -> Debug build
REM =====================================

set MODE=Release

if "%1"=="-d" set MODE=Debug

echo Build mode: %MODE%

set BUILD_DIR=%~dp0build_%MODE%

echo === Configure (%MODE%) ===
REM Pythonからpybind11のCMakeディレクトリを動的に取得する
FOR /F "tokens=*" %%i IN ('python -c "import pybind11; print(pybind11.get_cmake_dir().replace('\\', '/'))"') DO set PYBIND11_CMAKE_DIR=%%i

cmake -S "%~dp0." -B "%BUILD_DIR%" ^
    -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"

echo === Build (%MODE%) ===
cmake --build "%BUILD_DIR%" --config %MODE% --parallel

echo === Done (%MODE%) ===

endlocal
