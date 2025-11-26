#pragma once

#include <cassert>
#include <vector>

namespace tempest::anfwi {

class Grid2D {
public:
    Grid2D(int nx, int nz, float dx, float dz);

    float& operator()(int i, int k);
    float operator()(int i, int k) const;

    int nx() const { return nx_; }
    int nz() const { return nz_; }
    float dx() const { return dx_; }
    float dz() const { return dz_; }

private:
    int nx_;
    int nz_;
    float dx_;
    float dz_;
    std::vector<float> data_;
};

} // namespace tempest::anfwi
