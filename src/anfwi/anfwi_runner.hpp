#pragma once

#include <vector>

#include "tempest/common.hpp"
namespace tempest::anfwi {

class AnfwiSimulation {
public:
    explicit AnfwiSimulation(const SimulationConfig& cfg);

    SimulationStats run_headless();
    SimulationStats run_with_display();

    const SimulationConfig& config() const { return cfg_; }
    const std::vector<float>& pressure_field() const { return p_; }
    int steps_completed() const { return steps_completed_; }

private:
    SimulationStats run(bool verbose);
    void reset_fields();
    bool step_once();
    float current_peak() const;

    SimulationConfig cfg_;
    int nx_;
    int nz_;
    int n_;
    float dt2dx2_;
    int source_index_;
    int steps_completed_;

    std::vector<float> p_old_;
    std::vector<float> p_;
    std::vector<float> p_new_;
    std::vector<float> c2_;
};

SimulationStats run_anfwi_simulation(const SimulationConfig& cfg, bool use_display);

} // namespace tempest::anfwi
