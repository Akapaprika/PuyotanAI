cmake_minimum_required(VERSION 3.20)
project(PuyotanNative LANGUAGES CXX)

# ---------------------------------------------------------
# Global Settings (Standard C++23 and Include Path)
# ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Applying directory-level include path eliminates redundant target-level calls
include_directories(include)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# ---------------------------------------------------------
# Targets Definitions
# ---------------------------------------------------------

# 1. Core Engine (No Pybind, No Torch)
file(GLOB_RECURSE CORE_SOURCES src/core/*.cpp src/engine/*.cpp)
add_library(puyotan_core STATIC ${CORE_SOURCES})

# 2. Search Components (Beam Search, depends on Core only)
file(GLOB_RECURSE SEARCH_SOURCES src/search/*.cpp)
add_library(puyotan_search STATIC ${SEARCH_SOURCES})
target_link_libraries(puyotan_search PUBLIC puyotan_core)

# Benchmarks (C++ only)
file(GLOB_RECURSE BENCHMARK_SOURCES src/benchmarks/*.cpp)
add_executable(engine_benchmark src/benchmarks/engine_benchmark.cpp)
add_executable(beam_search_benchmark src/benchmarks/beam_search_benchmark.cpp)
target_link_libraries(engine_benchmark PRIVATE puyotan_core)
target_link_libraries(beam_search_benchmark PRIVATE puyotan_search)

# 3. Pybind11 Native Module
set(NATIVE_SOURCES src/bindings/bindings.cpp)
pybind11_add_module(puyotan_native ${NATIVE_SOURCES})
target_link_libraries(puyotan_native PRIVATE puyotan_core puyotan_search)

# ---------------------------------------------------------
# Compile & Link Options (All Targets)
# ---------------------------------------------------------
set(ALL_TARGETS 
    puyotan_core 
    puyotan_search 
    puyotan_native 
    engine_benchmark 
    beam_search_benchmark
)

if (MSVC)
    # Clear default flags to avoid warnings
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    foreach(T IN LISTS ALL_TARGETS)
        set_property(TARGET ${T} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        set_property(TARGET ${T} PROPERTY INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:TRUE>)
        
        # Unified PCH application
        target_precompile_headers(${T} PRIVATE include/puyotan/pch.hpp)
        
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # -------------------------------------------------------------
            # Clang-CL (LLVM) Peak Optimization Flags
            # -------------------------------------------------------------
            target_compile_options(${T} PRIVATE
                $<$<CONFIG:Debug>: /Od /Zi /RTC1 -Wno-everything >
                $<$<CONFIG:Release>:
                    /O2 /arch:AVX2 /fp:fast /DNDEBUG /FS
                    /clang:-O3
                    /clang:-march=znver1
                    /clang:-flto=thin
                    /clang:-fomit-frame-pointer
                    /clang:-fstrict-aliasing
                    /clang:-fno-math-errno
                    /clang:-funroll-loops
                    /clang:-fvectorize
                    /clang:-ffast-math
                    /clang:-mllvm /clang:-inline-threshold=600
                    -Wno-everything
                >
            )
        else()
            # -------------------------------------------------------------
            # MSVC (cl.exe) Optimization Flags
            # -------------------------------------------------------------
            target_compile_options(${T} PRIVATE
                /wd4267
                /wd4819
                $<$<CONFIG:Debug>: /Od /Zi /RTC1 /W4 /MP >
                $<$<CONFIG:Release>: /O2 /Ot /Oy /arch:AVX2 /Gy /Gw /GF /fp:fast /Oi /DNDEBUG /MP /FS /Ob3 /GS- /Qspectre- /favor:AMD64 /Zi >
            )
        endif()
    endforeach()

    # Disable Exception Handling and RTTI for non-Python target binaries to maximize inlining
    set(FAST_TARGETS puyotan_core puyotan_search engine_benchmark beam_search_benchmark)
    foreach(FT IN LISTS FAST_TARGETS)
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${FT} PRIVATE $<$<CONFIG:Release>: /GR- /EHs-c- >)
        else()
            target_compile_options(${FT} PRIVATE $<$<CONFIG:Release>: /GR- /EHs-c- /wd4530 >)
        endif()
    endforeach()

    # Re-enable Exception Handling for pybind11 module target
    target_compile_options(puyotan_native PRIVATE $<$<CONFIG:Release>: /EHsc >)

    # Linker Options (lld-link 用にクリーン化)
    set(LINK_TARGETS puyotan_native engine_benchmark beam_search_benchmark)
    foreach(LT IN LISTS LINK_TARGETS)
        target_link_options(${LT} PRIVATE
            $<$<CONFIG:Debug>: /DEBUG:FASTLINK >
            $<$<CONFIG:Release>: /DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO >
        )
        
        # Unified Post-build copy commands
        add_custom_command(TARGET ${LT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LT}> ${CMAKE_SOURCE_DIR}/dist/
        )
    endforeach()
else()
    # GCC / Native Clang (Linux / macOS)
    foreach(T IN LISTS ALL_TARGETS)
        set_property(TARGET ${T} PROPERTY INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:TRUE>)
        target_precompile_headers(${T} PRIVATE include/puyotan/pch.hpp)
        target_compile_options(${T} PRIVATE
            $<$<CONFIG:Release>: -O3 -flto=thin -march=native -ffast-math -funroll-loops -fomit-frame-pointer -DNDEBUG >
        )
    endforeach()
endif()