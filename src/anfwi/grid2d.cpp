#include "anfwi/grid2d.hpp"

namespace tempest::anfwi {

Grid2D::Grid2D(int nx, int nz, float dx, float dz)
    : nx_(nx), nz_(nz), dx_(dx), dz_(dz), data_(static_cast<size_t>(nx) * nz, 0.0f) {}

float& Grid2D::operator()(int i, int k) {
    assert(i >= 0 && i < nx_);
    assert(k >= 0 && k < nz_);
    return data_[static_cast<size_t>(k) + static_cast<size_t>(nx_) * i];
}

float Grid2D::operator()(int i, int k) const {
    assert(i >= 0 && i < nx_);
    assert(k >= 0 && k < nz_);
    return data_[static_cast<size_t>(k) + static_cast<size_t>(nx_) * i];
}

} // namespace tempest::anfwi
