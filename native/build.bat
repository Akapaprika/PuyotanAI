@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

:: =========================================================
:: build.bat  --  PuyotanAI native build script (Robust Edition)
:: =========================================================

set MODE=Release
set FORCE_COMPILER=
set PROFILING=OFF

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="-d"        set MODE=Debug
if "%~1"=="--debug"   set MODE=Debug
if "%~1"=="--msvc"    set FORCE_COMPILER=msvc
if "%~1"=="-msvc"     set FORCE_COMPILER=msvc
if "%~1"=="--clang"   set FORCE_COMPILER=clang
if "%~1"=="-clang"    set FORCE_COMPILER=clang
if "%~1"=="-p"        set PROFILING=ON
if "%~1"=="--prof"    set PROFILING=ON
if "%~1"=="--profile" set PROFILING=ON
shift
goto :parse_args
:args_done

:: pybind11 cmake dir
FOR /F "tokens=*" %%i IN ('python -c "import pybind11; print(pybind11.get_cmake_dir().replace(chr(92),chr(47)))"') DO set PYBIND11_CMAKE_DIR=%%i

set BUILD_DIR_VS=%~dp0build_%MODE%
set BUILD_DIR_NINJA=%~dp0build_%MODE%_clang

if "%PROFILING%"=="ON" (
    echo === Configure: %MODE% with AMD uProf Profiling Symbols ===
) else (
    echo === Configure: %MODE% - Pure Maximum Performance ===
)

:: ------------------------------------------------------------
:: Case 1: Force MSVC
:: ------------------------------------------------------------
if "%FORCE_COMPILER%"=="msvc" (
    echo [Build] Compiler: MSVC ^(forced^)
    cmake -S "%~dp0." -B "%BUILD_DIR_VS%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%" -DENABLE_PROFILING=%PROFILING%
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

        cmake -S "%~dp0." -B "%BUILD_DIR_NINJA%" -G "Ninja" -DCMAKE_BUILD_TYPE=%MODE% -DCMAKE_CXX_COMPILER="%CLANGCL_EXE%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%" -DENABLE_PROFILING=%PROFILING%
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
cmake -S "%~dp0." -B "%BUILD_DIR_VS%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%" -DENABLE_PROFILING=%PROFILING%
if errorlevel 1 ( exit /b 1 )
set BUILD_DIR=%BUILD_DIR_VS%

:build
echo === Build (%MODE%) ===
cmake --build "%BUILD_DIR%" --config %MODE% --parallel
if errorlevel 1 ( exit /b 1 )

echo === Done (%MODE%) ===
popd
endlocal
exit /b 0