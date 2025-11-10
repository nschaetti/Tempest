#include "tempest/init.hpp"

SimulationConfig load_config(const std::string& path) {
    try {
        YAML::Node node = YAML::LoadFile(path);
        SimulationConfig cfg;
        cfg.nx = node["nx"].as<int>();
        cfg.nz = node["nz"].as<int>();
        cfg.nt = node["nt"].as<int>();
        cfg.dx = node["dx"].as<float>();
        cfg.dt = node["dt"].as<float>();
        cfg.c0 = node["c0"].as<float>();
        cfg.block_size_x = node["block_size_x"].as<int>();
        cfg.block_size_y = node["block_size_y"].as<int>();
        cfg.display_scale = node["display_scale"].as<float>();
        cfg.display_interval = node["display_interval"].as<int>();
        return cfg;
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("Failed to load config: ") + ex.what());
    }
}

void print_config(const SimulationConfig& cfg) {
    std::cout << "Simulation settings:\n"
              << "  nx x nz        : " << cfg.nx << " x " << cfg.nz << "\n"
              << "  nt             : " << cfg.nt << "\n"
              << "  dx             : " << cfg.dx << " m\n"
              << "  dt             : " << cfg.dt << " s\n"
              << "  c0             : " << cfg.c0 << " m/s\n"
              << "  block size     : " << cfg.block_size_x << " x " << cfg.block_size_y << "\n"
              << "  display scale  : " << cfg.display_scale << "\n"
              << "  display interval: " << cfg.display_interval << std::endl;
}
