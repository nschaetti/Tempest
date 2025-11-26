#include "tempest/init.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

std::string trim(std::string_view text) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    std::size_t first = 0;
    while (first < text.size() && is_space(text[first])) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first && is_space(text[last - 1])) {
        --last;
    }
    return std::string{text.substr(first, last - first)};
}

template <typename T>
T parse_value(const std::unordered_map<std::string, std::string>& entries,
              const std::string& key,
              T default_value,
              bool required) {
    const auto it = entries.find(key);
    if (it == entries.end()) {
        if (required) {
            throw std::runtime_error("Missing required configuration key: " + key);
        }
        return default_value;
    }
    std::istringstream ss(it->second);
    T value{};
    ss >> value;
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse numeric value for key: " + key);
    }
    return value;
}

} // namespace

namespace tempest::anfwi {

SimulationConfig load_config(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open config file: " + path);
    }

    std::unordered_map<std::string, std::string> entries;
    std::string line;
    int line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto sep = line.find(':');
        if (sep == std::string::npos) {
            throw std::runtime_error("Invalid line " + std::to_string(line_no) +
                                     " in config file: missing ':' separator");
        }
        const std::string key = trim(line.substr(0, sep));
        const std::string value = trim(line.substr(sep + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("Invalid key/value pair on line " +
                                     std::to_string(line_no));
        }
        entries[key] = value;
    }

    SimulationConfig cfg;
    cfg.nx = parse_value<int>(entries, "nx", 0, true);
    cfg.nz = parse_value<int>(entries, "nz", 0, true);
    cfg.nt = parse_value<int>(entries, "nt", 0, true);
    cfg.dx = parse_value<float>(entries, "dx", 1.0f, true);
    cfg.dt = parse_value<float>(entries, "dt", 1.0f, true);
    cfg.c0 = parse_value<float>(entries, "c0", 1500.0f, false);
    cfg.block_size_x = parse_value<int>(entries, "block_size_x", 16, false);
    cfg.block_size_y = parse_value<int>(entries, "block_size_y", 16, false);
    cfg.display_scale = parse_value<float>(entries, "display_scale", 1.0f, false);
    cfg.display_interval = parse_value<int>(entries, "display_interval", 10, false);
    return cfg;
}

void print_config(const SimulationConfig& cfg) {
    std::cout << "Simulation settings:\n"
              << "  nx x nz         : " << cfg.nx << " x " << cfg.nz << "\n"
              << "  nt              : " << cfg.nt << "\n"
              << "  dx              : " << cfg.dx << " m\n"
              << "  dt              : " << cfg.dt << " s\n"
              << "  c0              : " << cfg.c0 << " m/s\n"
              << "  block size      : " << cfg.block_size_x << " x " << cfg.block_size_y << "\n"
              << "  display scale   : " << cfg.display_scale << "\n"
              << "  display interval: " << cfg.display_interval << std::endl;
}

} // namespace tempest::anfwi
