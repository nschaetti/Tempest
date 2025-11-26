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

namespace tempest::anfwi {

void initialize_wavefields(std::vector<float>& p_old,
                           std::vector<float>& p,
                           std::vector<float>& p_new,
                           std::vector<float>& c2,
                           float c0sq,
                           int source_index,
                           float amplitude);

void reset_wavefields(std::vector<float>& p_old,
                      std::vector<float>& p,
                      std::vector<float>& p_new,
                      int source_index,
                      float amplitude);

void wave_step(std::vector<float>& p_new,
               const std::vector<float>& p,
               const std::vector<float>& p_old,
               const std::vector<float>& c2,
               int nx,
               int nz,
               float dt2dx2);

} // namespace tempest::anfwi
