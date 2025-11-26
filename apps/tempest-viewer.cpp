#include "anfwi/anfwi_runner.hpp"
#include "tempest/init.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml> [--frames <n>]\n";
}

void print_heatmap(const std::vector<float>& field, int nx, int nz) {
    const std::string ramp = " .:-=+*#%@";
    const int rows = std::min(24, nz);
    const int cols = std::min(48, nx);
    float max_val = 0.0f;
    for (float v : field) {
        max_val = std::max(max_val, std::fabs(v));
    }
    if (max_val == 0.0f) {
        max_val = 1.0f;
    }

    for (int r = 0; r < rows; ++r) {
        int src_z = static_cast<int>((static_cast<double>(r) / rows) * nz);
        src_z = std::min(src_z, nz - 1);
        for (int c = 0; c < cols; ++c) {
            int src_x = static_cast<int>((static_cast<double>(c) / cols) * nx);
            src_x = std::min(src_x, nx - 1);
            std::size_t idx = static_cast<std::size_t>(src_z) * static_cast<std::size_t>(nx) +
                              static_cast<std::size_t>(src_x);
            const float norm = std::clamp(field[idx] / max_val, -1.0f, 1.0f);
            const std::size_t pos = static_cast<std::size_t>(
                (norm * 0.5f + 0.5f) * (ramp.size() - 1));
            std::cout << ramp[pos];
        }
        std::cout << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string config_path;
    int frames_override = -1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            frames_override = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (arg.starts_with("--")) {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        } else if (config_path.empty()) {
            config_path = arg;
        } else {
            std::cerr << "Unexpected positional argument: " << arg << "\n";
            return EXIT_FAILURE;
        }
    }

    if (config_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        auto cfg = tempest::anfwi::load_config(config_path);
        if (frames_override > 0) {
            cfg.nt = frames_override;
        }

        tempest::anfwi::AnfwiSimulation sim(cfg);
        const auto stats = sim.run_headless();
        std::cout << "Simulation completed in " << stats.elapsed_seconds << " s for "
                  << stats.steps << " steps. Peak amplitude " << stats.final_peak << "\n";
        print_heatmap(sim.pressure_field(), cfg.nx, cfg.nz);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
