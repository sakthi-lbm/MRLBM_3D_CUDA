#include <iostream>
#include "numerical_solution.cuh"

__device__ void numerical_solution_rhoeq_rotated(real unit_nx, real unit_ny,
                                                 const uint32_t incomingMask, const uint32_t outgoingMask,
                                                 const real ux_prime, const real uy_prime, const real uz_prime,
                                                 const real mxx_prime, const real myy_prime, const real mzz_prime,
                                                 const real myz_prime, real &rhoVar, real &ux, real &uy, real &uz,
                                                 real &mxx, real &myy, real &mzz, real &mxy, real &mxz, real &myz,
                                                 const int iter)
{
    const real cos_theta = unit_nx;
    const real sin_theta = unit_ny;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    real E_prime = toReal(0.0);
    real D_xy_prime = toReal(0.0);
    real D_xz_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F33_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);
    real F13_xy_prime = toReal(0.0);
    real F23_xy_prime = toReal(0.0);

    real F11_xz_prime = toReal(0.0);
    real F22_xz_prime = toReal(0.0);
    real F33_xz_prime = toReal(0.0);
    real F12_xz_prime = toReal(0.0);
    real F13_xz_prime = toReal(0.0);
    real F23_xz_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (int q = 0; q < Q; q++)
    {
        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cz = toReal(d_cz[q]);

        const real cx_prime = cx * cos_theta + cy * sin_theta;
        const real cy_prime = -cx * sin_theta + cy * cos_theta;
        const real cz_prime = cz;

        const real Hxx_prime = cx_prime * cx_prime - cs2;
        const real Hyy_prime = cy_prime * cy_prime - cs2;
        const real Hzz_prime = cz_prime * cz_prime - cs2;
        const real Hxy_prime = cx_prime * cy_prime;
        const real Hxz_prime = cx_prime * cz_prime;
        const real Hyz_prime = cy_prime * cz_prime;

        const real wq = d_w[q];
        const real common_factor = common_base_factor * wq;

        if (incomingMask & (1u << q))
        {
            // printf("incoming, q: %d \n", q);
            const real A_i = wq * (toReal(1.0) + as2 * (ux_prime * cx_prime + uy_prime * cy_prime +
                                                        uz_prime * cz_prime));
            const real E_i = A_i + common_factor * (ux_prime * ux_prime * Hxx_prime +
                                                    uy_prime * uy_prime * Hyy_prime +
                                                    uz_prime * uz_prime * Hzz_prime +
                                                    toReal(2.0) * ux_prime * uy_prime * Hxy_prime +
                                                    toReal(2.0) * ux_prime * uz_prime * Hxz_prime +
                                                    toReal(2.0) * uy_prime * uz_prime * Hyz_prime);

            E_prime += E_i;
            D_xy_prime += A_i * Hxy_prime;
            D_xz_prime += A_i * Hxz_prime;

            F11_xy_prime += (common_factor * Hxx_prime) * Hxy_prime;
            F22_xy_prime += (common_factor * Hyy_prime) * Hxy_prime;
            F33_xy_prime += (common_factor * Hzz_prime) * Hxy_prime;
            F12_xy_prime += (common_factor * Hxy_prime) * Hxy_prime;
            F13_xy_prime += (common_factor * Hxz_prime) * Hxy_prime;
            F23_xy_prime += (common_factor * Hyz_prime) * Hxy_prime;

            F11_xz_prime += (common_factor * Hxx_prime) * Hxz_prime;
            F22_xz_prime += (common_factor * Hyy_prime) * Hxz_prime;
            F33_xz_prime += (common_factor * Hzz_prime) * Hxz_prime;
            F12_xz_prime += (common_factor * Hxy_prime) * Hxz_prime;
            F13_xz_prime += (common_factor * Hxz_prime) * Hxz_prime;
            F23_xz_prime += (common_factor * Hyz_prime) * Hxz_prime;
        }
    }
    const real rhoI_prime = rhoVar;
    const real mxyI_prime = mxy;
    const real mxzI_prime = mxz;

    const real Rxy = E_prime * mxyI_prime - D_xy_prime;
    const real Rxz = E_prime * mxzI_prime - D_xz_prime;

    const real a1 = toReal(2.0) * F12_xy_prime;
    const real b1 = toReal(2.0) * F13_xy_prime;
    const real c1 = Rxy - mxx_prime * F11_xy_prime - myy_prime * F22_xy_prime - mzz_prime * F33_xy_prime -
                    toReal(2.0) * myz_prime * F23_xy_prime;

    const real a2 = toReal(2.0) * F12_xz_prime;
    const real b2 = toReal(2.0) * F13_xz_prime;
    const real c2 = Rxz - mxx_prime * F11_xz_prime - myy_prime * F22_xz_prime - mzz_prime * F33_xz_prime -
                    toReal(2.0) * myz_prime * F23_xz_prime;

    const real denominator = a2 * b1 - a1 * b2;
    const real inv_denom = toReal(1.0) / denominator;
    const real mxy_prime = (b1 * c2 - b2 * c1) * inv_denom;
    const real mxz_prime = (a2 * c1 - a1 * c2) * inv_denom;

    ux = ux_prime * cos_theta - uy_prime * sin_theta;
    uy = ux_prime * sin_theta + uy_prime * cos_theta;
    uz = uz_prime;

    mxx = mxx_prime * cos_theta * cos_theta + myy_prime * sin_theta * sin_theta - mxy_prime * sin_two_theta;
    myy = mxx_prime * sin_theta * sin_theta + myy_prime * cos_theta * cos_theta + mxy_prime * sin_two_theta;
    mzz = mzz_prime;
    mxy = (mxx_prime - myy_prime) * toReal(0.5) * sin_two_theta + mxy_prime * cos_two_theta;
    mxz = mxz_prime * cos_theta - myz_prime * sin_theta;
    myz = mxz_prime * sin_theta + myz_prime * cos_theta;

    rhoVar = rhoI_prime / E_prime;
}

__device__ void numerical_solution_strong_rotated(real unit_nx, real unit_ny,
                                                  const uint32_t incomingMask, const uint32_t outgoingMask,
                                                  const real ux_prime, const real uy_prime, const real uz_prime,
                                                  const real mxx_prime, const real myy_prime, const real mzz_prime,
                                                  const real myz_prime, real &rhoVar, real &ux, real &uy, real &uz,
                                                  real &mxx, real &myy, real &mzz, real &mxy, real &mxz, real &myz,
                                                  const int iter)
{
    const real cos_theta = unit_nx;
    const real sin_theta = unit_ny;
    const real sin_two_theta = toReal(2.0) * sin_theta * cos_theta;
    const real cos_two_theta = cos_theta * cos_theta - sin_theta * sin_theta;

    real A_prime = toReal(0.0);
    real E_prime = toReal(0.0);
    real B11_prime = toReal(0.0);
    real B22_prime = toReal(0.0);
    real B33_prime = toReal(0.0);
    real B12_prime = toReal(0.0);
    real B13_prime = toReal(0.0);
    real B23_prime = toReal(0.0);

    real D_xy_prime = toReal(0.0);
    real D_xz_prime = toReal(0.0);

    real F11_xy_prime = toReal(0.0);
    real F22_xy_prime = toReal(0.0);
    real F33_xy_prime = toReal(0.0);
    real F12_xy_prime = toReal(0.0);
    real F13_xy_prime = toReal(0.0);
    real F23_xy_prime = toReal(0.0);

    real F11_xz_prime = toReal(0.0);
    real F22_xz_prime = toReal(0.0);
    real F33_xz_prime = toReal(0.0);
    real F12_xz_prime = toReal(0.0);
    real F13_xz_prime = toReal(0.0);
    real F23_xz_prime = toReal(0.0);

    constexpr real common_base_factor = toReal(0.5) * as2 * as2;

    for (size_t q = 0; q < Q; q++)
    {
        const real cx_prime = toReal(d_cx[q]) * cos_theta + toReal(d_cy[q]) * sin_theta;
        const real cy_prime = -toReal(d_cx[q]) * sin_theta + toReal(d_cy[q]) * cos_theta;
        const real cz_prime = toReal(d_cz[q]);

        const real Hxx_prime = cx_prime * cx_prime - cs2;
        const real Hyy_prime = cy_prime * cy_prime - cs2;
        const real Hzz_prime = cz_prime * cz_prime - cs2;
        const real Hxy_prime = cx_prime * cy_prime;
        const real Hxz_prime = cx_prime * cz_prime;
        const real Hyz_prime = cy_prime * cz_prime;

        const real wq = d_w[q];
        const real common_factor = common_base_factor * wq;

        const real A_i = wq * (toReal(1.0) + as2 * (ux_prime * cx_prime + uy_prime * cy_prime +
                                                    uz_prime * cz_prime));
        if (outgoingMask & (1u << q))
        {
            A_prime += A_i;
            E_prime += A_i + common_factor * (ux_prime * ux_prime * Hxx_prime +
                                              uy_prime * uy_prime * Hyy_prime +
                                              uz_prime * uz_prime * Hzz_prime +
                                              toReal(2.0) * ux_prime * uy_prime * Hxy_prime +
                                              toReal(2.0) * ux_prime * uz_prime * Hxz_prime +
                                              toReal(2.0) * uy_prime * uz_prime * Hyz_prime);
            B11_prime += common_factor * Hxx_prime;
            B22_prime += common_factor * Hyy_prime;
            B33_prime += common_factor * Hzz_prime;
            B12_prime += common_factor * Hxy_prime;
            B13_prime += common_factor * Hxz_prime;
            B23_prime += common_factor * Hyz_prime;
        }

        if (incomingMask & (1u << q))
        {
            D_xy_prime += A_i * Hxy_prime;
            D_xz_prime += A_i * Hxz_prime;

            F11_xy_prime += (common_factor * Hxx_prime) * Hxy_prime;
            F22_xy_prime += (common_factor * Hyy_prime) * Hxy_prime;
            F33_xy_prime += (common_factor * Hzz_prime) * Hxy_prime;
            F12_xy_prime += (common_factor * Hxy_prime) * Hxy_prime;
            F13_xy_prime += (common_factor * Hxz_prime) * Hxy_prime;
            F23_xy_prime += (common_factor * Hyz_prime) * Hxy_prime;

            F11_xz_prime += (common_factor * Hxx_prime) * Hxz_prime;
            F22_xz_prime += (common_factor * Hyy_prime) * Hxz_prime;
            F33_xz_prime += (common_factor * Hzz_prime) * Hxz_prime;
            F12_xz_prime += (common_factor * Hxy_prime) * Hxz_prime;
            F13_xz_prime += (common_factor * Hxz_prime) * Hxz_prime;
            F23_xz_prime += (common_factor * Hyz_prime) * Hxz_prime;
        }
    }
    const real rhoI_prime = rhoVar;
    const real mxyI_prime = mxy;
    const real mxzI_prime = mxz;

    const real omega_term = toReal(1.0) - OMEGA;
    const real G_prime = omega_term * A_prime + OMEGA * E_prime;

    const real omega_mxy_term = omega_term * mxyI_prime;
    const real omega_mxz_term = omega_term * mxzI_prime;

    F11_xy_prime -= (omega_mxy_term * B11_prime);
    F22_xy_prime -= (omega_mxy_term * B22_prime);
    F33_xy_prime -= (omega_mxy_term * B33_prime);
    F12_xy_prime -= (omega_mxy_term * B12_prime);
    F13_xy_prime -= (omega_mxy_term * B13_prime);
    F23_xy_prime -= (omega_mxy_term * B23_prime);

    F11_xz_prime -= (omega_mxz_term * B11_prime);
    F22_xz_prime -= (omega_mxz_term * B22_prime);
    F33_xz_prime -= (omega_mxz_term * B33_prime);
    F12_xz_prime -= (omega_mxz_term * B12_prime);
    F13_xz_prime -= (omega_mxz_term * B13_prime);
    F23_xz_prime -= (omega_mxz_term * B23_prime);

    const real Rxy = G_prime * mxyI_prime - D_xy_prime;
    const real Rxz = G_prime * mxzI_prime - D_xz_prime;

    const real a1 = toReal(2.0) * F12_xy_prime;
    const real b1 = toReal(2.0) * F13_xy_prime;
    const real c1 = Rxy - mxx_prime * F11_xy_prime - myy_prime * F22_xy_prime - mzz_prime * F33_xy_prime -
                    toReal(2.0) * myz_prime * F23_xy_prime;

    const real a2 = toReal(2.0) * F12_xz_prime;
    const real b2 = toReal(2.0) * F13_xz_prime;
    const real c2 = Rxz - mxx_prime * F11_xz_prime - myy_prime * F22_xz_prime - mzz_prime * F33_xz_prime -
                    toReal(2.0) * myz_prime * F23_xz_prime;

    const real inv_denom = 1.0 / (a2 * b1 - a1 * b2);
    const real mxy_prime = (b1 * c2 - b2 * c1) * inv_denom;
    const real mxz_prime = (a2 * c1 - a1 * c2) * inv_denom;

    ux = ux_prime * cos_theta - uy_prime * sin_theta;
    uy = ux_prime * sin_theta + uy_prime * cos_theta;
    uz = uz_prime;

    mxx = mxx_prime * cos_theta * cos_theta + myy_prime * sin_theta * sin_theta - mxy_prime * sin_two_theta;
    myy = mxx_prime * sin_theta * sin_theta + myy_prime * cos_theta * cos_theta + mxy_prime * sin_two_theta;
    mzz = mzz_prime;
    mxy = (mxx_prime - myy_prime) * toReal(0.5) * sin_two_theta + mxy_prime * cos_two_theta;
    mxz = mxz_prime * cos_theta - myz_prime * sin_theta;
    myz = mxz_prime * sin_theta + myz_prime * cos_theta;

    const real denominator = G_prime + omega_term * (mxx_prime * B11_prime +
                                                     myy_prime * B22_prime +
                                                     mzz_prime * B33_prime +
                                                     toReal(2.0) * mxy_prime * B12_prime +
                                                     toReal(2.0) * mxz_prime * B13_prime +
                                                     toReal(2.0) * myz_prime * B23_prime);
    rhoVar = rhoI_prime / denominator;
}
