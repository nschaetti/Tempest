#include "anfwi/anfwi_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "tempest/simulation.hpp"

namespace tempest::anfwi {

AnfwiSimulation::AnfwiSimulation(const SimulationConfig& cfg)
    : cfg_(cfg),
      nx_(cfg.nx),
      nz_(cfg.nz),
      n_(cfg.nx * cfg.nz),
      dt2dx2_((cfg.dx != 0.0f) ? (cfg.dt * cfg.dt) / (cfg.dx * cfg.dx) : 0.0f),
      source_index_(((cfg.nz / 2) * cfg.nx) + (cfg.nx / 2)),
      steps_completed_(0),
      p_old_(static_cast<std::size_t>(cfg.nx) * static_cast<std::size_t>(cfg.nz)),
      p_(p_old_.size()),
      p_new_(p_old_.size()),
      c2_(p_old_.size()) {
    if (nx_ <= 0 || nz_ <= 0 || cfg_.nt <= 0) {
        throw std::runtime_error("Simulation dimensions and iteration counts must be positive");
    }
    if (cfg_.dx <= 0.0f || cfg_.dt <= 0.0f) {
        throw std::runtime_error("dx and dt must be strictly positive");
    }
    initialize_wavefields(p_old_, p_, p_new_, c2_, cfg_.c0 * cfg_.c0, source_index_, 1.0f);
}

SimulationStats AnfwiSimulation::run_headless() {
    return run(false);
}

SimulationStats AnfwiSimulation::run_with_display() {
    return run(true);
}

SimulationStats AnfwiSimulation::run(bool verbose) {
    reset_fields();
    SimulationStats stats{};
    const auto start = std::chrono::steady_clock::now();
    while (step_once()) {
        if (verbose && cfg_.display_interval > 0 &&
            (steps_completed_ % cfg_.display_interval == 0)) {
            std::cout << "Step " << steps_completed_ << " / " << cfg_.nt
                      << " peak amplitude " << current_peak() << std::endl;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    stats.steps = steps_completed_;
    stats.elapsed_seconds = elapsed.count();
    stats.final_peak = current_peak();
    if (verbose) {
        std::cout << "Simulation finished with " << stats.steps << " steps. "
                  << "Elapsed time: " << std::fixed << std::setprecision(3)
                  << stats.elapsed_seconds << " s. Peak amplitude "
                  << stats.final_peak << std::endl;
    }
    return stats;
}

void AnfwiSimulation::reset_fields() {
    reset_wavefields(p_old_, p_, p_new_, source_index_, 1.0f);
    steps_completed_ = 0;
}

bool AnfwiSimulation::step_once() {
    if (steps_completed_ >= cfg_.nt) {
        return false;
    }

    wave_step(p_new_, p_, p_old_, c2_, nx_, nz_, dt2dx2_);
    std::swap(p_old_, p_);
    std::swap(p_, p_new_);
    ++steps_completed_;
    return true;
}

float AnfwiSimulation::current_peak() const {
    if (p_.empty()) {
        return 0.0f;
    }
    const auto [min_it, max_it] = std::minmax_element(p_.begin(), p_.end());
    return std::max(std::fabs(*min_it), std::fabs(*max_it));
}

SimulationStats run_anfwi_simulation(const SimulationConfig& cfg, bool use_display) {
    AnfwiSimulation sim(cfg);
    return use_display ? sim.run_with_display() : sim.run_headless();
}

} // namespace tempest::anfwi
