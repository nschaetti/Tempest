cmake_minimum_required(VERSION 3.21)

# ============================================================
# Project configuration
# ============================================================
project(TempestCUDA
        LANGUAGES CXX CUDA
        DESCRIPTION "High-performance CUDA simulator for acoustic and elastic wave propagation (ANFWI)"
)

# ============================================================
# General settings
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable separate compilation and device linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# Build type and optimization flags
# ============================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3 -lineinfo")

# Optional: limit GPU architecture (adjust to your GPU)
# Example for RTX 3090 / compute capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

# ============================================================
# Include directories
# ============================================================
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================
# Source files
# ============================================================
file(GLOB_RECURSE SRC_FILES
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# ============================================================
# Executable
# ============================================================
add_executable(tempest_cuda ${SRC_FILES})

# Optional: change binary output directory
set_target_properties(tempest_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

# ============================================================
# Warnings & diagnostics
# ============================================================
target_compile_options(tempest_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler -Wall>
)

# ============================================================
# Summary (for debug)
# ============================================================
message(STATUS "============================================================")
message(STATUS "Tempest-CUDA build configuration")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler:       ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Output binary:       ${PROJECT_SOURCE_DIR}/bin/tempest_cuda")
message(STATUS "============================================================")
