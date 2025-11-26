#include "anfwi/anfwi_runner.hpp"
#include "tempest/init.hpp"

using tempest::anfwi::run_anfwi_simulation;

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml> [--show-display]\n"
              << "  --show-display  Print textual progress updates while running.\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string config_path;
    bool verbose_display = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--show-display") {
            verbose_display = true;
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
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (config_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        auto cfg = tempest::anfwi::load_config(config_path);
        tempest::anfwi::print_config(cfg);
        const auto stats = run_anfwi_simulation(cfg, verbose_display);
        std::cout << "Completed " << stats.steps << " steps in "
                  << stats.elapsed_seconds << " s. Peak amplitude "
                  << stats.final_peak << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
