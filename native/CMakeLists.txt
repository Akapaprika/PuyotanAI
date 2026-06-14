cmake_minimum_required(VERSION 3.20)
project(PuyotanNative LANGUAGES CXX)

# ---------------------------------------------------------
# Global Settings (Standard C++23 and Include Path)
# ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Applying directory-level include path eliminates 5 redundant target-level calls
include_directories(include)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------
# Targets Definitions
# ---------------------------------------------------------

# 1. Core Engine (No Pybind, No Torch)
file(GLOB_RECURSE CORE_SOURCES src/core/*.cpp src/engine/*.cpp)
add_library(puyotan_core STATIC ${CORE_SOURCES})
target_link_libraries(puyotan_core PUBLIC OpenMP::OpenMP_CXX)

# 2. Search Components (Beam Search, depends on Core only)
file(GLOB_RECURSE SEARCH_SOURCES src/search/*.cpp)
add_library(puyotan_search STATIC ${SEARCH_SOURCES})
target_link_libraries(puyotan_search PUBLIC puyotan_core OpenMP::OpenMP_CXX)

# Benchmarks (C++ only)
file(GLOB_RECURSE BENCHMARK_SOURCES src/benchmarks/*.cpp)
add_executable(engine_benchmark src/benchmarks/engine_benchmark.cpp)
add_executable(beam_search_benchmark src/benchmarks/beam_search_benchmark.cpp)
target_link_libraries(engine_benchmark PRIVATE puyotan_core OpenMP::OpenMP_CXX)
target_link_libraries(beam_search_benchmark PRIVATE puyotan_search OpenMP::OpenMP_CXX)

# 3. Pybind11 Native Module
file(GLOB_RECURSE NATIVE_SOURCES src/bindings/*.cpp)
pybind11_add_module(puyotan_native ${NATIVE_SOURCES})
target_link_libraries(puyotan_native PRIVATE puyotan_core puyotan_search OpenMP::OpenMP_CXX)

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
    # Clear default exception handling flag (/EHsc) to avoid D9025 command-line override warnings
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Ob2" "/Ob3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    foreach(T IN LISTS ALL_TARGETS)
        set_property(TARGET ${T} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        set_property(TARGET ${T} PROPERTY INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:TRUE>)
        
        # Unified PCH application in 1 line
        target_precompile_headers(${T} PRIVATE include/puyotan/pch.hpp)
        
        target_compile_options(${T} PRIVATE
            /wd4267
            /wd4819
            $<$<CONFIG:Debug>: /Od /Zi /RTC1 /W4 /MP >
            # /Z7: embed debug info in .obj to avoid PDB write-lock during /MP parallel compilation
            # /GF: string pooling to reduce binary size and improve cache locality
            # /favor:AMD64: tune instruction scheduling for AMD64 (benefits AMD 3020e)
            $<$<CONFIG:Release>: /O2 /Ot /Oy /arch:AVX2 /Gy /Gw /GF /fp:fast /Oi /DNDEBUG /MP /Ob3 /GS- /Qspectre- /favor:AMD64 >
        )
    endforeach()

    # Disable Exception Handling and RTTI for non-Python target binaries to maximize inlining
    set(FAST_TARGETS puyotan_core puyotan_search engine_benchmark beam_search_benchmark)
    foreach(FT IN LISTS FAST_TARGETS)
        target_compile_options(${FT} PRIVATE $<$<CONFIG:Release>: /GR- /EHs-c- /wd4530 >)
    endforeach()

    # Re-enable Exception Handling for pybind11 module target
    target_compile_options(puyotan_native PRIVATE $<$<CONFIG:Release>: /EHsc >)

    # Linker Options
    # Release: No /DEBUG to maximize LTO link-time and eliminate PDB overhead.
    #          /OPT:REF removes unreferenced functions; /OPT:ICF folds identical COMDATs.
    # Debug:   /DEBUG:FASTLINK for fast incremental link (no full PDB merge).
    set(LINK_TARGETS puyotan_native engine_benchmark beam_search_benchmark)
    foreach(LT IN LISTS LINK_TARGETS)
        target_link_options(${LT} PRIVATE
            $<$<CONFIG:Debug>: /DEBUG:FASTLINK >
            $<$<CONFIG:Release>: /OPT:REF /OPT:ICF /INCREMENTAL:NO >
        )
        
        # Unified Post-build copy commands to prevent code replication
        add_custom_command(TARGET ${LT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LT}> ${CMAKE_SOURCE_DIR}/dist/
        )
    endforeach()
endif()