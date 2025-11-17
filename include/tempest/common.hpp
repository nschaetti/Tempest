/*
 * This file is part of Tempest.
 *
 * Tempest is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Tempest is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Tempest.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

// Central include shared by both host-side C++ code and CUDA kernels.
// It collects third-party headers and common helper macros so every module
// can focus on its own logic without repeating boilerplate.

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Default CUDA tile sizes used by the wave propagation kernel. They can be
// overridden at compile time but act as a safety limit in main.cu.
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 32
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 32
#endif

// CUDA error checking helper. Novices often forget to inspect cudaError_t, so
// we fail fast with a descriptive message.
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                               \
    do {                                                                               \
        cudaError_t err__ = (expr);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                   \
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (false)
#endif

// OpenGL equivalent of CUDA_CHECK to keep GPU/GL state valid.
#ifndef GL_CHECK
#define GL_CHECK(expr)                                                                 \
    do {                                                                               \
        expr;                                                                          \
        GLenum gl_err__ = glGetError();                                                \
        if (gl_err__ != GL_NO_ERROR) {                                                 \
            std::cerr << "OpenGL error: 0x" << std::hex << gl_err__ << std::dec        \
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (false)
#endif

/**
 * @brief Configuration values for the whole simulation.
 *
 * All members are populated from the YAML file before any GPU memory or OpenGL
 * resources are created, so keep this struct POD-friendly.
 */
struct SimulationConfig {
    int nx = 0;                 ///< Number of columns (x direction) in the grid.
    int nz = 0;                 ///< Number of rows (z direction) in the grid.
    int nt = 0;                 ///< Number of time steps to run before stopping.
    float dx = 1.0f;            ///< Physical spacing between neighbouring cells.
    float dt = 0.001f;          ///< Time increment used per simulation iteration.
    float c0 = 1500.0f;         ///< Reference wave velocity, used to build c^2.
    int block_size_x = 16;      ///< CUDA block dimension along x for wave kernel.
    int block_size_y = 16;      ///< CUDA block dimension along z for wave kernel.
    float display_scale = 1.0f; ///< Factor applied to convert grid size to pixels.
    int display_interval = 10;  ///< Steps between OpenGL texture updates.
};

/**
 * @brief Mutable state shared between GLFW callbacks and the render loop.
 */
struct InputState {
    bool paused = false;        ///< True when user pressed 'P'.
    bool request_reset = false; ///< Set one frame after user pressed 'R'.
};
