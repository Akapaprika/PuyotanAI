@echo off
setlocal

REM === 引数解析 ===
set MODE=Release

if "%1"=="-d" set MODE=Debug
if "%1"=="--debug" set MODE=Debug
if "%1"=="-r" set MODE=RelWithDebInfo
if "%1"=="--relwithdebinfo" set MODE=RelWithDebInfo

echo Build mode: %MODE%

set BUILD_DIR=build_%MODE%

REM === Release のときだけ完全クリーン ===
if "%MODE%"=="Release" (
    if exist %BUILD_DIR% (
        echo === Clean ===
        rmdir /s /q %BUILD_DIR%
    )
)

echo === Configure %MODE% ===
cmake -S . -B %BUILD_DIR% -Dpybind11_DIR=C:\Users\FMV\AppData\Local\Programs\Python\Python314\Lib\site-packages\pybind11\share\cmake\pybind11

echo === Build %MODE% ===
cmake --build %BUILD_DIR% --config %MODE% -- /m

echo === Done (%MODE%) ===
endlocal
