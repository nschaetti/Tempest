#include "tempest/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tempest::anfwi {

namespace {

void ensure_same_size(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("wavefield buffers must have matching sizes");
    }
}

} // namespace

void initialize_wavefields(std::vector<float>& p_old,
                           std::vector<float>& p,
                           std::vector<float>& p_new,
                           std::vector<float>& c2,
                           float c0sq,
                           int source_index,
                           float amplitude) {
    if (p_old.size() != p.size() || p.size() != p_new.size() || p.size() != c2.size()) {
        throw std::runtime_error("wavefield buffers must share the same length");
    }

    std::fill(p_old.begin(), p_old.end(), 0.0f);
    std::fill(p.begin(), p.end(), 0.0f);
    std::fill(p_new.begin(), p_new.end(), 0.0f);
    std::fill(c2.begin(), c2.end(), c0sq);

    if (source_index >= 0 && source_index < static_cast<int>(p.size())) {
        p[source_index] = amplitude;
    }
}

void reset_wavefields(std::vector<float>& p_old,
                      std::vector<float>& p,
                      std::vector<float>& p_new,
                      int source_index,
                      float amplitude) {
    ensure_same_size(p_old, p);
    ensure_same_size(p, p_new);

    std::fill(p_old.begin(), p_old.end(), 0.0f);
    std::fill(p.begin(), p.end(), 0.0f);
    std::fill(p_new.begin(), p_new.end(), 0.0f);

    if (source_index >= 0 && source_index < static_cast<int>(p.size())) {
        p[source_index] = amplitude;
    }
}

void wave_step(std::vector<float>& p_new,
               const std::vector<float>& p,
               const std::vector<float>& p_old,
               const std::vector<float>& c2,
               int nx,
               int nz,
               float dt2dx2) {
    if (nx <= 2 || nz <= 2) {
        throw std::runtime_error("grid size needs to be larger than the stencil radius");
    }

    const std::size_t expected_size = static_cast<std::size_t>(nx) * static_cast<std::size_t>(nz);
    if (p.size() != expected_size || p_old.size() != expected_size ||
        p_new.size() != expected_size || c2.size() != expected_size) {
        throw std::runtime_error("wave_step received buffers that do not match nx * nz");
    }

    auto idx = [nx](int x, int z) {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(x);
    };

    for (int iz = 1; iz < nz - 1; ++iz) {
        for (int ix = 1; ix < nx - 1; ++ix) {
            const std::size_t center = idx(ix, iz);
            const float laplacian =
                p[idx(ix - 1, iz)] + p[idx(ix + 1, iz)] +
                p[idx(ix, iz - 1)] + p[idx(ix, iz + 1)] -
                4.0f * p[center];

            p_new[center] = 2.0f * p[center] - p_old[center] + c2[center] * dt2dx2 * laplacian;
        }
    }

    // Copy the border directly to avoid accidentally reading outside the grid.
    for (int ix = 0; ix < nx; ++ix) {
        p_new[idx(ix, 0)] = p[idx(ix, 0)];
        p_new[idx(ix, nz - 1)] = p[idx(ix, nz - 1)];
    }
    for (int iz = 0; iz < nz; ++iz) {
        p_new[idx(0, iz)] = p[idx(0, iz)];
        p_new[idx(nx - 1, iz)] = p[idx(nx - 1, iz)];
    }
}

} // namespace tempest::anfwi
