@echo off
setlocal

:: =========================================================
:: build.bat  --  PuyotanAI native build script
::
:: Usage:
::   build.bat [options]
::   -d / --debug    Debug build  (default: Release)
::   --msvc          Force MSVC compiler
::   --clang         Force Clang compiler
::
:: Priority (auto-detection):
::   1. Standalone LLVM clang-cl (winget) via Ninja
::   2. VS-integrated ClangCL toolset (-T ClangCL)
::   3. Default MSVC toolset (fallback)
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
:: Case 2: Standalone clang-cl (winget LLVM) + Ninja
:: ------------------------------------------------------------
set CLANGCL_EXE=
where clang-cl >nul 2>&1
if not errorlevel 1 (
    set CLANGCL_EXE=clang-cl
) else (
    if exist "C:\Program Files\LLVM\bin\clang-cl.exe" (
        set CLANGCL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
    )
)

set NINJA_EXE=
where ninja >nul 2>&1
if not errorlevel 1 ( set NINJA_EXE=ninja )
if "%NINJA_EXE%"=="" (
    FOR /F "tokens=*" %%N IN ('echo %USERPROFILE%\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe') DO (
        if exist "%%N" set NINJA_EXE=%%N
    )
)
if "%NINJA_EXE%"=="" (
    if exist "C:\ProgramData\chocolatey\bin\ninja.exe" set "NINJA_EXE=C:\ProgramData\chocolatey\bin\ninja.exe"
)

if not "%CLANGCL_EXE%"=="" if not "%NINJA_EXE%"=="" (
    :: Setup MSVC environment for clang-cl (SDK header/library paths)
    set VCVARS=
    if "%VCVARS%"=="" if exist "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"   set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"   set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files\Microsoft Visual Studio\17\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"   set "VCVARS=C:\Program Files\Microsoft Visual Studio\17\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files\Microsoft Visual Studio\17\Professional\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\17\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files\Microsoft Visual Studio\17\Community\VC\Auxiliary\Build\vcvarsall.bat"   set "VCVARS=C:\Program Files\Microsoft Visual Studio\17\Community\VC\Auxiliary\Build\vcvarsall.bat"
    if "%VCVARS%"=="" if exist "C:\Program Files (x86)\Microsoft Visual Studio\17\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\17\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

    if not "%VCVARS%"=="" (
        call "%VCVARS%" x64 >nul 2>&1
    )

    echo [Build] Compiler: Clang ^(LLVM clang-cl^) + Ninja
    cmake -S "%~dp0." -B "%BUILD_DIR_NINJA%" ^
        -G "Ninja" ^
        -DCMAKE_BUILD_TYPE=%MODE% ^
        -DCMAKE_C_COMPILER="%CLANGCL_EXE%" ^
        -DCMAKE_CXX_COMPILER="%CLANGCL_EXE%" ^
        -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
        -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%"
    if not errorlevel 1 (
        set BUILD_DIR=%BUILD_DIR_NINJA%
        goto :build
    )
    echo [Notice] Ninja+clang-cl failed. Falling back...
)

:: ------------------------------------------------------------
:: Case 3: VS-integrated ClangCL (-T ClangCL)
:: ------------------------------------------------------------
cmake -S "%~dp0." -B "%BUILD_DIR_VS%" -T ClangCL -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%" >nul 2>&1
if not errorlevel 1 (
    echo [Build] Compiler: Clang ^(VS integrated ClangCL^)
    set BUILD_DIR=%BUILD_DIR_VS%
    goto :build
)

:: Clean up if VS ClangCL failed
if exist "%BUILD_DIR_VS%\CMakeCache.txt" del /f /q "%BUILD_DIR_VS%\CMakeCache.txt" >nul 2>&1
if exist "%BUILD_DIR_VS%\CMakeFiles"     rmdir /s /q "%BUILD_DIR_VS%\CMakeFiles" >nul 2>&1

if "%FORCE_COMPILER%"=="clang" (
    echo [Error] --clang requested but no usable Clang installation found.
    exit /b 1
)

:: ------------------------------------------------------------
:: Case 4: MSVC Fallback
:: ------------------------------------------------------------
echo [Build] Compiler: MSVC ^(fallback^)
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
