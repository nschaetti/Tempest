#pragma once

#include <cstddef>

#include "anfwi/grid2d.hpp"

namespace tempest::anfwi {

struct SimulationState {
    int nx;
    int nz;
    float dx;
    float dz;
    float dt;
    int nt;
    Grid2D p_old;
    Grid2D p;
    Grid2D p_new;
    Grid2D c;
    Grid2D damping;
    int sx;
    int sz;
    float f0;

    SimulationState(int nx_, int nz_, float dx_, float dz_, int nt_)
        : nx(nx_),
          nz(nz_),
          dx(dx_),
          dz(dz_),
          dt(0.0f),
          nt(nt_),
          p_old(nx_, nz_, dx_, dz_),
          p(nx_, nz_, dx_, dz_),
          p_new(nx_, nz_, dx_, dz_),
          c(nx_, nz_, dx_, dz_),
          damping(nx_, nz_, dx_, dz_),
          sx(nx_ / 2),
          sz(nz_ / 2),
          f0(10.0f) {}
};

} // namespace tempest::anfwi
