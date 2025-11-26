cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0148 NEW)

project(Tempest
        LANGUAGES CXX
        DESCRIPTION "CPU-friendly scientific simulators (ANFWI, SurfDisp, etc.)"
)

option(BUILD_PYTHON "Build Python Bindings" ON)
option(BUILD_TOOLS "Build standalone tools" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type (Debug, Release)" FORCE)
endif()

set(TEMPEST_SRC
        src/anfwi/grid2d.cpp
        src/anfwi/init.cpp
        src/anfwi/simulation.cpp
        src/anfwi/anfwi_runner.cpp
        src/surfdisp.cpp
)

add_library(tempest STATIC ${TEMPEST_SRC})
target_include_directories(tempest
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_options(tempest PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

if(BUILD_TOOLS)
    add_subdirectory(apps)
endif()

if(BUILD_PYTHON)
    add_subdirectory(python/pytempest)
endif()

message(STATUS "============================================================")
message(STATUS "Tempest build configuration")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "Output binaries:     ${PROJECT_SOURCE_DIR}/bin/")
message(STATUS "============================================================")
