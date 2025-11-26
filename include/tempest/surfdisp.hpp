#pragma once

#include <array>

namespace surfdisp {

inline constexpr int NL = 100;
inline constexpr int NP = 200;

using ArrNL = std::array<double, NL>;
using ArrNP = std::array<double, NP>;

void surfdisp96(const ArrNL& thkm,
                const ArrNL& vpm,
                const ArrNL& vsm,
                const ArrNL& rhom,
                int nlayer,
                int iflsph,
                int iwave,
                int mode,
                int igr,
                int kmax,
                const ArrNP& t,
                ArrNP& cg,
                int& err);

} // namespace surfdisp
