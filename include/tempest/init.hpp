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
