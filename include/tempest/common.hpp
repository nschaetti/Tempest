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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Configuration values for the whole simulation.
 *
 * The structure intentionally stays plain so it can be copied across the C++
 * library, CLI tools and Python bindings without additional glue code.
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

namespace tempest::anfwi {

struct SimulationStats {
    int steps = 0;
    double elapsed_seconds = 0.0;
    float final_peak = 0.0f;
};

} // namespace tempest::anfwi
