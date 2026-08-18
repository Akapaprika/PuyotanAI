@echo off
setlocal enabledelayedexpansion

:: =========================================================
:: build.bat  --  PuyotanAI native build script (Robust Edition)
:: =========================================================

set MODE=Release
set FORCE_COMPILER=

if "%1"=="-d"      set MODE=Debug
if "%1"=="--debug" set MODE=Debug
if "%2"=="-d"      set MODE=Debug
if "%2"=="--debug" set MODE=Debug

if "%1"=="--msvc"  set FORCE_COMPILER=msvc
if "%1"=="-msvc"   set FORCE_COMPILER=msvc
if "%2"=="--msvc"  set FORCE_COMPILER=msvc
if "%2"=="-msvc"   set FORCE_COMPILER=msvc

if "%1"=="--clang" set FORCE_COMPILER=clang
if "%1"=="-clang"  set FORCE_COMPILER=clang
if "%2"=="--clang" set FORCE_COMPILER=clang
if "%2"=="-clang"  set FORCE_COMPILER=clang

:: pybind11 cmake dir
FOR /F "tokens=*" %%i IN ('python -c "import pybind11; print(pybind11.get_cmake_dir().replace(chr(92),chr(47)))"') DO set PYBIND11_CMAKE_DIR=%%i

set BUILD_DIR_VS=%~dp0build_%MODE%
set BUILD_DIR_NINJA=%~dp0build_%MODE%_clang

echo === Configure (%MODE%) ===

:: ------------------------------------------------------------
:: Case 1: Force MSVC
:: ------------------------------------------------------------
if "%FORCE_COMPILER%"=="msvc" (
    echo [Build] Compiler: MSVC ^(forced^)
    cmake -S "%~dp0." -B "%BUILD_DIR_VS%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"
    if errorlevel 1 ( exit /b 1 )
    set BUILD_DIR=%BUILD_DIR_VS%
    goto :build
)

:: ------------------------------------------------------------
:: Case 2: Clang + Ninja
:: ------------------------------------------------------------
set "LLVM_DIR=C:/Program Files/LLVM/bin"
if exist "C:\Program Files\LLVM\bin\clang-cl.exe" (
    set "PATH=C:\Program Files\LLVM\bin;%PATH%"
    set "CLANGCL_EXE=C:/Program Files/LLVM/bin/clang-cl.exe"
) else (
    where clang-cl >nul 2>&1
    if not errorlevel 1 ( set "CLANGCL_EXE=clang-cl" )
)

if not "%CLANGCL_EXE%"=="" (
    :: VS 環境変数のロード
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
                call "%%i\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
            )
        )
    )

    where ninja >nul 2>&1
    if not errorlevel 1 (
        echo [Build] Compiler: Clang ^(LLVM clang-cl^) + Ninja
        if exist "%BUILD_DIR_NINJA%\CMakeCache.txt" del /f /q "%BUILD_DIR_NINJA%\CMakeCache.txt" >nul 2>&1

        cmake -S "%~dp0." -B "%BUILD_DIR_NINJA%" ^
            -G "Ninja" ^
            -DCMAKE_BUILD_TYPE=%MODE% ^
            -DCMAKE_CXX_COMPILER="%CLANGCL_EXE%" ^
            -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"
        if not errorlevel 1 (
            set BUILD_DIR=%BUILD_DIR_NINJA%
            goto :build
        )
        echo [Notice] Ninja failed. Falling back to Visual Studio...
    )
)

:: ------------------------------------------------------------
:: Case 3: Visual Studio Fallback (100% 確実に成功するルート)
:: ------------------------------------------------------------
echo [Build] Compiler: MSVC ^(Visual Studio^)
cmake -S "%~dp0." -B "%BUILD_DIR_VS%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"
if errorlevel 1 ( exit /b 1 )
set BUILD_DIR=%BUILD_DIR_VS%

:build
echo === Build (%MODE%) ===
cmake --build "%BUILD_DIR%" --config %MODE% --parallel
if errorlevel 1 ( exit /b 1 )

echo === Done (%MODE%) ===
endlocal
exit /b 0