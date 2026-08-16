@echo off
setlocal

set MODE=Release
set TOOLSET=-T ClangCL

if "%1"=="-d" set MODE=Debug
if "%1"=="-msvc" set TOOLSET=
if "%1"=="--msvc" set TOOLSET=
if "%1"=="-clang" set TOOLSET=-T ClangCL
if "%1"=="--clang" set TOOLSET=-T ClangCL

if "%2"=="-d" set MODE=Debug
if "%2"=="-msvc" set TOOLSET=
if "%2"=="--msvc" set TOOLSET=
if "%2"=="-clang" set TOOLSET=-T ClangCL
if "%2"=="--clang" set TOOLSET=-T ClangCL

echo [Build] Mode: %MODE%, Toolset: %TOOLSET%

set BUILD_DIR=%~dp0build_%MODE%

echo === Configure (%MODE%) ===
FOR /F "tokens=*" %%i IN ('python -c "import pybind11; print(pybind11.get_cmake_dir().replace('\\', '/'))"') DO set PYBIND11_CMAKE_DIR=%%i

if "%TOOLSET%"=="" (
    cmake -S "%~dp0." -B "%BUILD_DIR%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"
) else (
    cmake -S "%~dp0." -B "%BUILD_DIR%" %TOOLSET% -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"
)

echo === Build (%MODE%) ===
cmake --build "%BUILD_DIR%" --config %MODE% --parallel

echo === Done (%MODE%) ===

endlocal
