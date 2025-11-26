#include "tempest/surfdisp.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--kmax <samples>] [--mode <n>]\n"
              << "Computes Rayleigh-wave group velocities for a built-in layered model.\n";
}

} // namespace

int main(int argc, char** argv) {
    int kmax = 50;
    int mode = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kmax" && i + 1 < argc) {
            kmax = std::atoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (kmax <= 0 || kmax > surfdisp::NP) {
        std::cerr << "kmax must be between 1 and " << surfdisp::NP << "\n";
        return EXIT_FAILURE;
    }
    if (mode <= 0) {
        std::cerr << "Mode must be positive.\n";
        return EXIT_FAILURE;
    }

    surfdisp::ArrNL thkm{};
    surfdisp::ArrNL vpm{};
    surfdisp::ArrNL vsm{};
    surfdisp::ArrNL rhom{};

    thkm[0] = 5.0;
    thkm[1] = 10.0;
    thkm[2] = 0.0;

    vpm[0] = 5.5;
    vpm[1] = 6.2;
    vpm[2] = 6.8;

    vsm[0] = 3.2;
    vsm[1] = 3.6;
    vsm[2] = 4.0;

    rhom[0] = 2.5;
    rhom[1] = 2.7;
    rhom[2] = 2.8;

    surfdisp::ArrNP periods{};
    for (int i = 0; i < kmax; ++i) {
        periods[i] = 0.5 + (static_cast<double>(i) / std::max(1, kmax - 1)) * 15.0;
    }

    surfdisp::ArrNP cg{};
    int err = 0;
    surfdisp::surfdisp96(thkm, vpm, vsm, rhom, 3, 0, 2, mode, 1, kmax, periods, cg, err);

    if (err != 0) {
        std::cerr << "surfdisp96 finished with error code " << err << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Rayleigh group velocities (mode " << mode << ")\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < kmax; ++i) {
        std::cout << "T = " << std::setw(6) << periods[i]
                  << " s, cg = " << std::setw(7) << cg[i] << " km/s\n";
    }

    return EXIT_SUCCESS;
}
