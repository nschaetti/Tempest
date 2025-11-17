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

#include "tempest/common.hpp"

/**
 * @brief Perform one explicit finite-difference time-domain (FDTD) update.
 *
 * Each CUDA thread updates a single grid cell by applying the 5-point Laplacian
 * stencil to the current pressure field `p`. Shared memory is used to cache a
 * tile of the field, which dramatically reduces global memory traffic.
 */
__global__ void wave_step_kernel(
    float* p_new,
    const float* p,
    const float* p_old,
    const float* c2,
    int nx,
    int nz,
    float dt2dx2
);

/**
 * @brief Fill device buffers with initial values and inject a starting pulse.
 *
 * The reference velocity (c0) is squared and copied into d_c2, while all wave
 * fields are zeroed except for the source cell.
 */
void initialize_wavefields(float* d_p_old,
                           float* d_p,
                           float* d_p_new,
                           float* d_c2,
                           float c0sq,
                           int n,
                           int source_index,
                           float amplitude);

/**
 * @brief Reset the three pressure fields and re-apply the point source.
 *
 * Called whenever the user presses 'R' so the animation restarts without
 * reallocating GPU memory.
 */
void reset_wavefields(float* d_p_old,
                      float* d_p,
                      float* d_p_new,
                      int n,
                      int source_index,
                      float amplitude);
