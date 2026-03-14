cmake_minimum_required(VERSION 3.20)
project(PuyotanNative LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

pybind11_add_module(puyotan_native ${SOURCES})

target_compile_features(puyotan_native PRIVATE cxx_std_20)
set_target_properties(puyotan_native PROPERTIES CXX_EXTENSIONS OFF)
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

    add_custom_command(TARGET puyotan_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_SOURCE_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:puyotan_native>
        ${CMAKE_SOURCE_DIR}/dist/
    )

endif()
