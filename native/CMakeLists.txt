cmake_minimum_required(VERSION 3.20)
project(PuyotanNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES
    src/*.cpp
)

pybind11_add_module(puyotan_native ${SOURCES})

target_include_directories(puyotan_native PRIVATE include)

# ===== MSVC 最適化設定 =====
if (MSVC)
    target_compile_options(puyotan_native PRIVATE /utf-8)
    set_property(TARGET puyotan_native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )

    # Debug
    target_compile_options(puyotan_native PRIVATE
        $<$<CONFIG:Debug>:
            /Od
            /Zi
            /RTC1
            /W4
        >
    )

    target_link_options(puyotan_native PRIVATE
        $<$<CONFIG:Debug>:
            /DEBUG
        >
    )

    # Release
    target_compile_options(puyotan_native PRIVATE
        $<$<CONFIG:Release>:
            /O2
            /GL
            /Oi
            /Ot
            /fp:fast
            /arch:AVX2
            /DNDEBUG
        >
    )

    target_link_options(puyotan_native PRIVATE
        $<$<CONFIG:Release>:
            /LTCG
            /OPT:REF
            /OPT:ICF
            /INCREMENTAL:NO
        >
    )

    # RelWithDebInfo（研究用メイン）
    target_compile_options(puyotan_native PRIVATE
        $<$<CONFIG:RelWithDebInfo>:
            /O2
            /Zi
            /GL
        >
    )

    target_link_options(puyotan_native PRIVATE
        $<$<CONFIG:RelWithDebInfo>:
            /DEBUG
            /LTCG
            /INCREMENTAL:NO
        >
    )

    add_custom_command(TARGET puyotan_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_SOURCE_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:puyotan_native>
        ${CMAKE_SOURCE_DIR}/dist/
    )

endif()
