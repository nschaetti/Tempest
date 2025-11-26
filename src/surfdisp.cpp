#include "tempest/surfdisp.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <vector>

// A lightweight C++ translation of the surfdisp96 Fortran simulator.
// The implementation mirrors the structure of the consolidated Fortran
// source and the accompanying Python port, using fixed-size arrays to
// stay close to the original memory layout.

namespace surfdisp
{
    struct VarTerms
    {
        double w{};
        double cosp{};
        double exa{};
        double a0{};
        double cpcq{};
        double cpy{};
        double cpz{};
        double cqw{};
        double cqx{};
        double xy{};
        double xz{};
        double wy{};
        double wz{};
    };

    struct Matrix5x5 {
        std::array<std::array<double, 5>, 5> data{};
    };

    inline double copysign_double(double value, double sign_ref)
    {
        return std::copysign(value, sign_ref);
    }

    // Forward declarations
    double dltar(double wvno, double omega, int kk, const ArrNL &d, const ArrNL &a, const ArrNL &b,
                 const ArrNL &rho, const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                 int mmax, int llw, double twopi);

    // Transform spherical earth to flat earth (directly from the Fortran version).
    void sphere(int ifunc, int iflag, ArrNL &d, ArrNL &a, ArrNL &b, ArrNL &rho,
                ArrNL &rtp, ArrNL &dtp, ArrNL &btp, int mmax, int llw, double twopi) {
        (void)llw;
        (void)twopi;
        static double dhalf = 0.0;
        const double ar = 6370.0;
        double dr = 0.0;
        double r0 = ar;

        if (mmax <= 0 || mmax > NL) {
            throw std::runtime_error("sphere: invalid mmax");
        }

        // Keep index semantics close to Fortran by treating mmax as the count of layers (1-based).
        d[mmax] = 1.0;

        if (iflag == 0)
        {
            for (int i = 0; i < mmax; ++i)
            {
                dtp[i] = d[i];
                rtp[i] = rho[i];
            }
            for (int i = 0; i < mmax; ++i)
            {
                dr += d[i];
                double r1 = ar - dr;
                double z0 = ar * std::log(ar / r0);
                double z1 = ar * std::log(ar / r1);
                d[i] = z1 - z0;

                // Use layer midpoint scaling
                double tmp = (ar + ar) / (r0 + r1);
                a[i] *= tmp;
                b[i] *= tmp;
                btp[i] = tmp;
                r0 = r1;
            }
            dhalf = d[mmax];
        }
        else
        {
            d[mmax] = dhalf;
            for (int i = 0; i < mmax; ++i) {
                if (ifunc == 1) {
                    rho[i] = rtp[i] * std::pow(btp[i], -5.0);
                } else if (ifunc == 2) {
                    rho[i] = rtp[i] * std::pow(btp[i], -2.275);
                }
            }
        }

        d[mmax] = 0.0;
    }

    // Evaluate variables for the compound matrix.
    VarTerms var(double p, double q, double ra, double rb, double wvno, double xka, double xkb, double dpth)
    {
        VarTerms vt{};
        double pex = 0.0;
        double sex = 0.0;
        double cosp = 1.0;
        double w = 0.0;
        double x = 0.0;

        // P-wave eigenfunctions
        if (wvno < xka) {
            double sinp = std::sin(p);
            w = sinp / ra;
            x = -ra * sinp;
            cosp = std::cos(p);
        } else if (wvno == xka) {
            cosp = 1.0;
            w = dpth;
            x = 0.0;
        } else {
            pex = p;
            double fac = 0.0;
            if (p < 16.0) {
                fac = std::exp(-2.0 * p);
            }
            cosp = (1.0 + fac) * 0.5;
            double sinp = (1.0 - fac) * 0.5;
            w = sinp / ra;
            x = ra * sinp;
        }

        // S-wave eigenfunctions
        double cosq = 1.0;
        double y = 0.0;
        double z = 0.0;
        if (wvno < xkb) {
            double sinq = std::sin(q);
            y = sinq / rb;
            z = -rb * sinq;
            cosq = std::cos(q);
        } else if (wvno == xkb) {
            cosq = 1.0;
            y = dpth;
            z = 0.0;
        } else {
            sex = q;
            double fac = 0.0;
            if (q < 16.0) {
                fac = std::exp(-2.0 * q);
            }
            cosq = (1.0 + fac) * 0.5;
            double sinq = (1.0 - fac) * 0.5;
            y = sinq / rb;
            z = rb * sinq;
        }

        double exa = pex + sex;
        double a0 = 0.0;
        if (exa < 60.0)
        {
            a0 = std::exp(-exa);
        }
        double cpcq = cosp * cosq;
        double cpy = cosp * y;
        double cpz = cosp * z;
        double cqw = cosq * w;
        double cqx = cosq * x;
        double xy = x * y;
        double xz = x * z;
        double wy = w * y;
        double wz = w * z;
        double qmp = sex - pex;
        double fac = 0.0;
        if (qmp > -40.0) {
            fac = std::exp(qmp);
        }
        cosq *= fac;
        y *= fac;
        z *= fac;

        vt.w = w;
        vt.cosp = cosp;
        vt.exa = exa;
        vt.a0 = a0;
        vt.cpcq = cpcq;
        vt.cpy = cpy;
        vt.cpz = cpz;
        vt.cqw = cqw;
        vt.cqx = cqx;
        vt.xy = xy;
        vt.xz = xz;
        vt.wy = wy;
        vt.wz = wz;
        return vt;
    }

    // Normalize vectors to control over/underflow.
    std::pair<std::array<double, 5>, double> normc(const std::array<double, 5> &ee)
    {
        std::array<double, 5> out = ee;
        double t1 = 0.0;
        for (double v : out)
        {
            t1 = std::max(t1, std::abs(v));
        }
        if (t1 < 1.0e-40)
        {
            t1 = 1.0;
        }
        for (double &v : out)
        {
            v /= t1;
        }
        double ex = std::log(t1);
        return {out, ex};
    }

    // Calculate Dunkin's matrix for layered media.
    Matrix5x5 dnka(double wvno2, double gam, double gammk, double rho, double a0, double cpcq,
                   double cpy, double cpz, double cqw, double cqx, double xy, double xz,
                   double wy, double wz)
    {
        Matrix5x5 m{};
        const double one = 1.0;
        const double two = 2.0;
        double gamm1 = gam - one;
        double twgm1 = gam + gamm1;
        double gmgmk = gam * gammk;
        double gmgm1 = gam * gamm1;
        double gm1sq = gamm1 * gamm1;
        double rho2 = rho * rho;
        double a0pq = a0 - cpcq;

        auto &c = m.data;
        c[0][0] = cpcq - two * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy;
        c[0][1] = (wvno2 * cpy - cqx) / rho;
        c[0][2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho;
        c[0][3] = (cpz - wvno2 * cqw) / rho;
        c[0][4] = -(two * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2;

        c[1][0] = (gmgmk * cpz - gm1sq * cqw) * rho;
        c[1][1] = cpcq;
        c[1][2] = gammk * cpz - gamm1 * cqw;
        c[1][3] = -wz;
        c[1][4] = c[0][3];

        c[3][0] = (gm1sq * cpy - gmgmk * cqx) * rho;
        c[3][1] = -xy;
        c[3][2] = gamm1 * cpy - gammk * cqx;
        c[3][3] = c[1][1];
        c[3][4] = c[0][1];

        c[4][0] = -(two * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy) * rho2;
        c[4][1] = c[3][0];
        c[4][2] = -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy) * rho;
        c[4][3] = c[1][0];
        c[4][4] = c[0][0];

        double t = -two * wvno2;
        c[2][0] = t * c[4][2];
        c[2][1] = t * c[3][2];
        c[2][2] = a0 + two * (cpcq - c[0][0]);
        c[2][3] = t * c[1][2];
        c[2][4] = t * c[0][2];
        return m;
    }

    // Find SH dispersion values (Love waves).
    double dltar1(double wvno, double omega, const ArrNL &d, const ArrNL &a, const ArrNL &b,
                  const ArrNL &rho, const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                  int mmax, int llw, double twopi)
    {
        (void)a;
        (void)rtp;
        (void)dtp;
        (void)btp;
        (void)twopi;
        if (mmax <= 0) {
            return 0.0;
        }
        const int halfspace = mmax - 1;
        int lower = std::max(0, llw - 1);

        double beta1 = b[halfspace];
        double rho1 = rho[halfspace];
        double xkb = omega / beta1;
        double wvnop = wvno + xkb;
        double wvnom = std::abs(wvno - xkb);
        double rb = std::sqrt(wvnop * wvnom);
        double e1 = rho1 * rb;
        double e2 = 1.0 / (beta1 * beta1);

        for (int m = halfspace - 1; m >= lower; --m)
        {
            beta1 = b[m];
            rho1 = rho[m];
            double xmu = rho1 * beta1 * beta1;
            xkb = omega / beta1;
            wvnop = wvno + xkb;
            wvnom = std::abs(wvno - xkb);
            rb = std::sqrt(wvnop * wvnom);
            double q = d[m] * rb;

            double y = 0.0;
            double z = 0.0;
            double cosq = 1.0;
            if (wvno < xkb)
            {
                double sinq = std::sin(q);
                y = sinq / rb;
                z = -rb * sinq;
                cosq = std::cos(q);
            }
            else if (wvno == xkb)
            {
                cosq = 1.0;
                y = d[m];
                z = 0.0;
            }
            else
            {
                double fac = 0.0;
                if (q < 16.0) fac = std::exp(-2.0 * q);
                cosq = (1.0 + fac) * 0.5;
                double sinq = (1.0 - fac) * 0.5;
                y = sinq / rb;
                z = rb * sinq;
            }

            double e10 = e1 * cosq + e2 * xmu * z;
            double e20 = e1 * y / xmu + e2 * cosq;
            double xnor = std::abs(e10);
            double ynor = std::abs(e20);

            if (ynor > xnor) xnor = ynor;
            if (xnor < 1.0e-40) xnor = 1.0;

            e1 = e10 / xnor;
            e2 = e20 / xnor;
        }
        return e1;
    }

    // Find P-SV dispersion values (Rayleigh waves).
    double dltar4(double wvno, double omega, const ArrNL &d, const ArrNL &a, const ArrNL &b,
                  const ArrNL &rho, const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                  int mmax, int llw, double twopi)
    {
        (void)rtp;
        (void)dtp;
        (void)btp;
        (void)twopi;
        if (mmax <= 0) {
            return 0.0;
        }
        if (omega < 1.0e-4) {
            omega = 1.0e-4;
        }
        double wvno2 = wvno * wvno;
        const int halfspace = mmax - 1;
        int lower = std::max(0, llw - 1);

        double xka = omega / a[halfspace];
        double xkb = omega / b[halfspace];
        double wvnop = wvno + xka;
        double wvnom = std::abs(wvno - xka);
        double ra = std::sqrt(wvnop * wvnom);
        wvnop = wvno + xkb;
        wvnom = std::abs(wvno - xkb);
        double rb = std::sqrt(wvnop * wvnom);
        double t = b[halfspace] / omega;

        double gammk = 2.0 * t * t;
        double gam = gammk * wvno2;
        double gamm1 = gam - 1.0;
        double rho1 = rho[halfspace];

        std::array<double, 5> e{};
        e[0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb);
        e[1] = -rho1 * ra;
        e[2] = rho1 * (gamm1 - gammk * ra * rb);
        e[3] = rho1 * rb;
        e[4] = wvno2 - ra * rb;

        for (int m = halfspace - 1; m >= lower; --m)
        {
            xka = omega / a[m];
            xkb = omega / b[m];
            t = b[m] / omega;
            gammk = 2.0 * t * t;
            gam = gammk * wvno2;
            wvnop = wvno + xka;
            wvnom = std::abs(wvno - xka);
            ra = std::sqrt(wvnop * wvnom);
            wvnop = wvno + xkb;
            wvnom = std::abs(wvno - xkb);
            rb = std::sqrt(wvnop * wvnom);
            double dpth = d[m];
            rho1 = rho[m];
            double p = ra * dpth;
            double q = rb * dpth;

            VarTerms vt = var(p, q, ra, rb, wvno, xka, xkb, dpth);
            Matrix5x5 ca = dnka(wvno2, gam, gammk, rho1, vt.a0, vt.cpcq, vt.cpy, vt.cpz,
                                vt.cqw, vt.cqx, vt.xy, vt.xz, vt.wy, vt.wz);

            std::array<double, 5> ee{};
            for (int i = 0; i < 5; ++i) {
                double cr = 0.0;
                for (int j = 0; j < 5; ++j) {
                    cr += e[j] * ca.data[j][i];
                }
                ee[i] = cr;
            }
            auto norm = normc(ee);
            e = norm.first;
        }

        if (llw != 1)
        {
            xka = omega / a[0];
            wvnop = wvno + xka;
            wvnom = std::abs(wvno - xka);
            ra = std::sqrt(wvnop * wvnom);
            double dpth = d[0];
            rho1 = rho[0];
            double p = ra * dpth;
            double znul = 1.0e-5;
            VarTerms vt = var(p, znul, ra, znul, wvno, xka, znul, dpth);
            double w0 = -rho1 * vt.w;
            return vt.cosp * e[0] + w0 * e[1];
        }
        return e[0];
    }

    double dltar(double wvno, double omega, int kk, const ArrNL &d, const ArrNL &a, const ArrNL &b,
                 const ArrNL &rho, const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                 int mmax, int llw, double twopi)
    {
        if (kk == 1)
        {
            return dltar1(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
        }
        return dltar4(wvno, omega, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
    }

    // Starting solution for phase velocity calculation.
    double gtsolh(double a, double b)
    {
        if (a <= 0.0 || b <= 0.0 || b > a)
        {
            throw std::runtime_error("gtsolh: invalid velocities");
        }
        double c = 0.95 * b;
        for (int i = 0; i < 5; ++i)
        {
            double gamma = b / a;
            double kappa = c / b;
            double k2 = kappa * kappa;
            double gk2 = (gamma * kappa) * (gamma * kappa);
            double fac1 = std::sqrt(1.0 - gk2);
            double fac2 = std::sqrt(1.0 - k2);
            double fr = (2.0 - k2) * (2.0 - k2) - 4.0 * fac1 * fac2;
            double frp = -4.0 * (2.0 - k2) * kappa +
                         4.0 * fac2 * gamma * gamma * kappa / fac1 +
                         4.0 * fac1 * kappa / fac2;
            frp /= b;
            c = c - fr / frp;
        }
        return c;
    }

    // Interval halving method.
    std::pair<double, double> half(double c1, double c2, double omega, int ifunc,
                                   const ArrNL &d, const ArrNL &a, const ArrNL &b, const ArrNL &rho,
                                   const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                                   int mmax, int llw, double twopi)
    {
        double c3 = 0.5 * (c1 + c2);
        double wvno = omega / c3;
        double del3 = dltar(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
        return {c3, del3};
    }

    // Hybrid method for refining root once it has been bracketed.
    double nevill(double t, double c1, double c2, double del1, double del2, int ifunc,
                  const ArrNL &d, const ArrNL &a, const ArrNL &b, const ArrNL &rho,
                  const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                  int mmax, int llw, double twopi)
    {
        double omega = 2.0 * M_PI / t;
        auto h = half(c1, c2, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
        double c3 = h.first;
        double del3 = h.second;
        int nev = 1;
        int nctrl = 1;
        std::array<double, 20> x{};
        std::array<double, 20> y{};

        int m = 0;
        while (true) {
            nctrl++;
            if (nctrl >= 100) {
                break;
            }
            if (c3 < std::min(c1, c2) || c3 > std::max(c1, c2)) {
                nev = 0;
                auto halved = half(c1, c2, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
                c3 = halved.first;
                del3 = halved.second;
            }

            double s13 = del1 - del3;
            double s32 = del3 - del2;

            if (std::copysign(1.0, del3) * std::copysign(1.0, del1) < 0.0) {
                c2 = c3;
                del2 = del3;
            } else {
                c1 = c3;
                del1 = del3;
            }

            if (std::abs(c1 - c2) <= 1e-6 * c1) {
                break;
            }

            if (std::copysign(1.0, s13) != std::copysign(1.0, s32)) {
                nev = 0;
            }

            double ss1 = std::abs(del1);
            double s1 = 0.01 * ss1;
            double ss2 = std::abs(del2);
            double s2 = 0.01 * ss2;

            if (s1 > ss2 || s2 > ss1 || nev == 0) {
                auto halved = half(c1, c2, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
                c3 = halved.first;
                del3 = halved.second;
                nev = 1;
                m = 1;
            } else {
                if (nev == 2) {
                    x[m] = c3;
                    y[m] = del3;
                } else {
                    x[0] = c1;
                    y[0] = del1;
                    x[1] = c2;
                    y[1] = del2;
                    m = 1;
                }

                bool fallback = false;
                for (int kk = 1; kk <= m; ++kk) {
                    int j = m - kk;
                    double denom = y[m] - y[j];
                    if (std::abs(denom) < 1.0e-10 * std::abs(y[m])) {
                        fallback = true;
                        break;
                    }
                    x[j] = (-y[j] * x[j + 1] + y[m] * x[j]) / denom;
                }
                if (fallback) {
                    auto halved = half(c1, c2, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
                    c3 = halved.first;
                    del3 = halved.second;
                    nev = 1;
                    m = 1;
                } else {
                    c3 = x[0];
                    double wvno = omega / c3;
                    del3 = dltar(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
                    nev = 2;
                    m = std::min(m + 1, 10);
                }
            }
        }
        return c3;
    }

    // Bracket dispersion curve and refine it.
    std::pair<double, int> getsol(double t1, double c1, double clow, double dc, double cm, double betmx,
                                  int ifunc, int ifirst, const ArrNL &d, const ArrNL &a, const ArrNL &b,
                                  const ArrNL &rho, const ArrNL &rtp, const ArrNL &dtp, const ArrNL &btp,
                                  int mmax, int llw) {
        static double del1st = 0.0;
        const double twopi = 2.0 * M_PI;
        double omega = twopi / t1;
        double wvno = omega / c1;

        double del1 = dltar(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);

        if (ifirst == 1) {
            del1st = del1;
        }
        double plmn = copysign_double(1.0, del1st) * std::copysign(1.0, del1);
        int idir = 1;
        if (ifirst != 1 && plmn < 0.0) {
            idir = -1;
        }

        while (true) {
            double c2 = (idir > 0) ? c1 + dc : c1 - dc;

            if (c2 <= clow) {
                idir = 1;
                c1 = clow;
                continue;
            }

            omega = twopi / t1;
            wvno = omega / c2;
            double del2 = dltar(wvno, omega, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
            if (std::copysign(1.0, del1) != std::copysign(1.0, del2)) {
                double cn = nevill(t1, c1, c2, del1, del2, ifunc, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
                c1 = cn;
                if (c1 > betmx) {
                    return {c1, -1};
                }
                return {c1, 1};
            }

            c1 = c2;
            del1 = del2;

            if (c1 < cm || c1 >= (betmx + dc)) {
                return {c1, -1};
            }
        }
    }

    // Main entry point mirroring the Fortran subroutine.
    void surfdisp96(const ArrNL &thkm, const ArrNL &vpm, const ArrNL &vsm, const ArrNL &rhom,
                    int nlayer, int iflsph, int iwave, int mode, int igr, int kmax,
                    const ArrNP &t, ArrNP &cg, int &err) {
        ArrNL d{};
        ArrNL a{};
        ArrNL b{};
        ArrNL rho{};
        ArrNL rtp{};
        ArrNL dtp{};
        ArrNL btp{};
        ArrNP c{};
        ArrNP cb{};
        std::array<int, 3> iverb{};

        int mmax = nlayer;
        int nsph = iflsph;
        err = 0;

        for (int i = 0; i < mmax; ++i) {
            b[i] = vsm[i];
            a[i] = vpm[i];
            d[i] = thkm[i];
            rho[i] = rhom[i];
        }

        int idispl = 0;
        int idispr = 0;
        if (iwave == 1) {
            idispl = kmax;
        } else if (iwave == 2) {
            idispr = kmax;
        } else {
            throw std::runtime_error("iwave must be 1 (Love) or 2 (Rayleigh)");
        }

        iverb[1] = 0;
        iverb[2] = 0;

        const double sone0 = 1.5;
        const double ddc0 = 0.005;
        const double h0 = 0.005;

        int llw = 1;
        if (b[0] <= 0.0) {
            llw = 2;
        }
        const double twopi = 2.0 * M_PI;
        const double one = 1.0e-2;
        if (nsph == 1) {
            sphere(0, 0, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
        }

        int jmn = 0;
        double betmx = -1.0e20;
        double betmn = 1.0e20;
        int jsol = 1;
        for (int i = 0; i < mmax; ++i) {
            if (b[i] > 0.01 && b[i] < betmn) {
                betmn = b[i];
                jmn = i;
                jsol = 1;
            } else if (b[i] <= 0.01 && a[i] < betmn) {
                betmn = a[i];
                jmn = i;
                jsol = 0;
            }
            if (b[i] > betmx) {
                betmx = b[i];
            }
        }

        for (int ifunc = 1; ifunc <= 2; ++ifunc) {
            if (ifunc == 1 && idispl <= 0) continue;
            if (ifunc == 2 && idispr <= 0) continue;

            if (nsph == 1) {
                sphere(ifunc, 1, d, a, b, rho, rtp, dtp, btp, mmax, llw, twopi);
            }

            double ddc = ddc0;
            double sone = sone0;
            double h = h0;

            if (sone < 0.01) sone = 2.0;
            double onea = sone;

            double cc1 = 0.0;
            if (jsol == 0) {
                cc1 = betmn;
            } else {
                cc1 = gtsolh(a[jmn], b[jmn]);
            }
            cc1 = 0.95 * cc1;
            cc1 = 0.90 * cc1;
            double cc = cc1;
            double dc = std::abs(ddc);
            double c1 = cc;
            double cm = cc;

            for (int i = 0; i < kmax; ++i) {
                cb[i] = 0.0;
                c[i] = 0.0;
            }

            int ift = 999;
            for (int iq = 1; iq <= mode; ++iq) {
                int is = 0;
                int ie = kmax;

                int k = is;
                for (; k < ie; ++k) {
                    if (k >= ift) break;

                    double t1 = t[k];
                    double t1a = t1;
                    double t1b = t1;
                    if (igr > 0) {
                        t1a = t1 / (1.0 + h);
                        t1b = t1 / (1.0 - h);
                        t1 = t1a;
                    }

                    int ifirst = 0;
                    double clow = 0.0;
                    if (k == is && iq == 1) {
                        c1 = cc;
                        clow = cc;
                        ifirst = 1;
                    } else if (k == is && iq > 1) {
                        c1 = c[is] + one * dc;
                        clow = c1;
                        ifirst = 1;
                    } else if (k > is && iq > 1) {
                        ifirst = 0;
                        clow = c[k] + one * dc;
                        c1 = c[k - 1];
                        if (c1 < clow) c1 = clow;
                    } else if (k > is && iq == 1) {
                        ifirst = 0;
                        c1 = c[k - 1] - onea * dc;
                        clow = cm;
                    } else {
                        throw std::runtime_error("Unexpected search branch");
                    }

                    auto sol = getsol(t1, c1, clow, dc, cm, betmx, ifunc, ifirst,
                                      d, a, b, rho, rtp, dtp, btp, mmax, llw);
                    c1 = sol.first;
                    int iret = sol.second;
                    if (iret == -1) break;
                    c[k] = c1;

                    if (igr > 0) {
                        t1 = t1b;
                        ifirst = 0;
                        clow = cb[k] + one * dc;
                        c1 = c1 - onea * dc;
                        auto sol2 = getsol(t1, c1, clow, dc, cm, betmx, ifunc, ifirst,
                                           d, a, b, rho, rtp, dtp, btp, mmax, llw);
                        c1 = sol2.first;
                        iret = sol2.second;
                        if (iret == -1) {
                            c1 = c[k];
                        }
                        cb[k] = c1;
                    } else {
                        c1 = 0.0;
                    }

                    double cc0 = c[k];
                    if (igr == 0) {
                        cg[k] = cc0;
                    } else {
                        double gvel = (1.0 / t1a - 1.0 / t1b) /
                                      (1.0 / (t1a * cc0) - 1.0 / (t1b * c1));
                        cg[k] = gvel;
                    }
                }

                ift = k;
            }
        }
    }
} // namespace surfdisp
