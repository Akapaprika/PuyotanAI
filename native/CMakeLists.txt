cmake_minimum_required(VERSION 3.20)
project(PuyotanNative LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ===== ONNX Runtime =====
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
set(ORT_INCLUDE "${ORT_DIR}/include")
set(ORT_LIB     "${ORT_DIR}/lib/onnxruntime.lib")
set(ORT_DLL     "${ORT_DIR}/lib/onnxruntime.dll")

# Include all source files in the new structure
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

pybind11_add_module(puyotan_native ${SOURCES})

target_compile_features(puyotan_native PRIVATE cxx_std_23)
set_target_properties(puyotan_native PROPERTIES CXX_EXTENSIONS OFF)

# Use unified include directory + ONNX Runtime headers
target_include_directories(puyotan_native PRIVATE include ${ORT_INCLUDE})
target_link_libraries(puyotan_native PRIVATE OpenMP::OpenMP_CXX ${ORT_LIB})

# ===== MSVC Optimization Settings =====
if (MSVC)
    target_compile_options(puyotan_native PRIVATE /utf-8)
    set_property(TARGET puyotan_native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )

    # Debug
    target_compile_options(puyotan_native PRIVATE
        $<$<CONFIG:Debug>: /Od /Zi /RTC1 /W4 >
    )
    target_link_options(puyotan_native PRIVATE
        $<$<CONFIG:Debug>: /DEBUG >
    )

    # Release
    target_compile_options(puyotan_native PRIVATE
        $<$<CONFIG:Release>: /O2 /GL /arch:AVX2 /Ob2 /Gy /Gw /fp:fast /DNDEBUG >
    )
    target_link_options(puyotan_native PRIVATE
        $<$<CONFIG:Release>: /LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO >
    )

    # Post-build: copy pyd + onnxruntime.dll to dist/
    add_custom_command(TARGET puyotan_native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:puyotan_native> ${CMAKE_SOURCE_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" ${CMAKE_SOURCE_DIR}/dist/
    )
endif()
