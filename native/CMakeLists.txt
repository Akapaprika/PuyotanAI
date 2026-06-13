cmake_minimum_required(VERSION 3.20)
project(PuyotanNative LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ---------------------------------------------------------
# Targets
# ---------------------------------------------------------

# 1. Core Engine (No Pybind, No Torch)
file(GLOB_RECURSE CORE_SOURCES
    src/core/*.cpp
    src/engine/*.cpp
)
add_library(puyotan_core STATIC ${CORE_SOURCES})
target_compile_features(puyotan_core PRIVATE cxx_std_23)
target_include_directories(puyotan_core PUBLIC include)
target_link_libraries(puyotan_core PUBLIC OpenMP::OpenMP_CXX)

# 2. Search Components (Beam Search, future MCTS — depends on Core only)
file(GLOB_RECURSE SEARCH_SOURCES
    src/search/*.cpp
)
add_library(puyotan_search STATIC ${SEARCH_SOURCES})
target_compile_features(puyotan_search PRIVATE cxx_std_23)
target_include_directories(puyotan_search PUBLIC include)
target_link_libraries(puyotan_search PUBLIC puyotan_core OpenMP::OpenMP_CXX)

# Benchmarks (C++ only)
file(GLOB_RECURSE BENCHMARK_SOURCES
    src/benchmarks/*.cpp
)
add_executable(engine_benchmark src/benchmarks/engine_benchmark.cpp)
add_executable(beam_search_benchmark src/benchmarks/beam_search_benchmark.cpp)
target_compile_features(engine_benchmark PRIVATE cxx_std_23)
target_compile_features(beam_search_benchmark PRIVATE cxx_std_23)
target_include_directories(engine_benchmark PRIVATE include)
target_include_directories(beam_search_benchmark PRIVATE include)
target_link_libraries(engine_benchmark PRIVATE puyotan_core OpenMP::OpenMP_CXX)
target_link_libraries(beam_search_benchmark PRIVATE puyotan_search OpenMP::OpenMP_CXX)

# 3. Pybind11 Native Module
file(GLOB_RECURSE NATIVE_SOURCES
    src/bindings/*.cpp
)
pybind11_add_module(puyotan_native ${NATIVE_SOURCES})
target_compile_features(puyotan_native PRIVATE cxx_std_23)
target_include_directories(puyotan_native PRIVATE include)
target_link_libraries(puyotan_native PRIVATE puyotan_core puyotan_search OpenMP::OpenMP_CXX)
target_precompile_headers(puyotan_native PRIVATE include/puyotan/pch.hpp)

# ---------------------------------------------------------
# Compile Options (All Targets)
# ---------------------------------------------------------
set(ALL_TARGETS 
    puyotan_core 
    puyotan_search 
    puyotan_native 
    engine_benchmark 
    beam_search_benchmark
)

if (MSVC)
    string(REPLACE "/Ob2" "/Ob3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    foreach(T IN LISTS ALL_TARGETS)
        set_property(TARGET ${T} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        
        target_compile_options(${T} PRIVATE
            /wd4267
            /wd4819
            $<$<CONFIG:Debug>: /Od /Zi /RTC1 /W4 /MP >
            $<$<CONFIG:Release>: /O2 /Ot /arch:AVX2 /Gy /Gw /fp:fast /Oi /DNDEBUG /MP /GL /Ob3 /GS- >
        )
        # Enable Link-Time Optimization (LTO / Whole Program Optimization)
        set_property(TARGET ${T} PROPERTY INTERPROCEDURAL_OPTIMIZATION $<$<CONFIG:Release>:TRUE>)
    endforeach()

    set(LINK_TARGETS puyotan_native engine_benchmark beam_search_benchmark)
    foreach(LT IN LISTS LINK_TARGETS)
        target_link_options(${LT} PRIVATE
            $<$<CONFIG:Debug>: /DEBUG >
            $<$<CONFIG:Release>: /OPT:REF /OPT:ICF /LTCG /INCREMENTAL:NO >
        )
    endforeach()

    # Post-build copy to dist
    add_custom_command(TARGET puyotan_native POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:puyotan_native> ${CMAKE_SOURCE_DIR}/dist/
    )

    add_custom_command(TARGET engine_benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:engine_benchmark> ${CMAKE_SOURCE_DIR}/dist/
    )

    add_custom_command(TARGET beam_search_benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:beam_search_benchmark> ${CMAKE_SOURCE_DIR}/dist/
    )
endif()