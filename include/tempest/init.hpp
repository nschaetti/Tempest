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
 * @brief Read the YAML configuration file and populate SimulationConfig.
 *
 * @throws std::runtime_error when any entry is missing or the file is unreadable.
 */
SimulationConfig load_config(const std::string& path);

/**
 * @brief Pretty-print the configuration so the user can verify the inputs.
 */
void print_config(const SimulationConfig& cfg);
