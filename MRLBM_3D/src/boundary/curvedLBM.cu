#include <iostream>

#include "curvedLBM.cuh"

__device__ void evaluate_incoming_moments_rotated(const real unit_nx, const real unit_ny,
                                                  const uint32_t incomingMask,
                                                  const real *pop, real &rho,
                                                  real &mxx, real &myy, real &mzz,
                                                  real &mxy, real &mxz, real &myz)
{
    const real cos_theta = unit_nx;
    const real sin_theta = unit_ny;

    real rhoI_prime = toReal(0.0);
    real mxxI_prime = toReal(0.0);
    real myyI_prime = toReal(0.0);
    real mzzI_prime = toReal(0.0);
    real mxyI_prime = toReal(0.0);
    real mxzI_prime = toReal(0.0);
    real myzI_prime = toReal(0.0);

    for (int q = 0; q < Q; q++)
    {
        if (incomingMask & (1u << q))
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

            rhoI_prime += pop[q];
            mxxI_prime += pop[q] * Hxx_prime;
            myyI_prime += pop[q] * Hyy_prime;
            mzzI_prime += pop[q] * Hzz_prime;
            mxyI_prime += pop[q] * Hxy_prime;
            mxzI_prime += pop[q] * Hxz_prime;
            myzI_prime += pop[q] * Hyz_prime;
        }
    }
    const real inv_rho = toReal(1.0) / rhoI_prime;

    rho = rhoI_prime;
    mxx = mxxI_prime * inv_rho;
    myy = myyI_prime * inv_rho;
    mzz = mzzI_prime * inv_rho;
    mxy = mxyI_prime * inv_rho;
    mxz = mxzI_prime * inv_rho;
    myz = myzI_prime * inv_rho;
}

__device__ void curved_boundary_condition_rotated(const real unit_nx, const real unit_ny, const real delta,
                                                    unsigned int xb, unsigned int yb, unsigned int zb,
                                                    const real xw, const real yw, const real zw,
                                                    const cylinderVar &cylinder,
                                                    const nodeType_t nodeType, const nodeVar &dMom,
                                                    real &rho, real &ux, real &uy, real &uz,
                                                    real &mxx, real &myy, real &mzz,
                                                    real &mxy, real &mxz, real &myz,
                                                    const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                                    const int NODE_TYPE, const real D_WALL, const int iter)
{
    // First reference fluid point
    const real x1 = xw + delx * unit_nx;
    const real y1 = yw + delx * unit_ny;

    // second reference fluid point
    const real x2 = xw + toReal(2.0) * delx * unit_nx;
    const real y2 = yw + toReal(2.0) * delx * unit_ny;

    real ux_prime, uy_prime, uz_prime;
    real mxx_prime, myy_prime, mzz_prime, myz_prime;
    {
        // Interpolation and rotation at fluid points
        VelocityMoments vm1 = initerpolate_and_rotate(unit_nx, unit_ny, x1, y1, zb, dMom);
        VelocityMoments vm2 = initerpolate_and_rotate(unit_nx, unit_ny, x2, y2, zb, dMom);

        // physical wall conditions
        VelocityMoments vmw;
        vmw.ux = UX_PRIME;
        vmw.uy = UY_PRIME;
        vmw.uz = UZ_PRIME;
        vmw.mxx = UX_PRIME * UX_PRIME;
        vmw.myy = UY_PRIME * UY_PRIME;
        vmw.mzz = UZ_PRIME * UZ_PRIME;
        vmw.myz = UY_PRIME * UZ_PRIME;

        ux_prime = extrapolation_quadratic(delta, vmw.ux, vm1.ux, vm2.ux);
        uy_prime = extrapolation_quadratic(delta, vmw.uy, vm1.uy, vm2.uy);
        uz_prime = extrapolation_quadratic(delta, vmw.uz, vm1.uz, vm2.uz);
        mxx_prime = extrapolation_quadratic(delta, vmw.mxx, vm1.mxx, vm2.mxx);
        myy_prime = extrapolation_quadratic(delta, vmw.myy, vm1.myy, vm2.myy);
        mzz_prime = extrapolation_quadratic(delta, vmw.mzz, vm1.mzz, vm2.mzz);
        myz_prime = extrapolation_quadratic(delta, vmw.myz, vm1.myz, vm2.myz);

        // ux_prime = vmw.ux;
        // uy_prime = vmw.uy;
        // uz_prime = vmw.uz;
        // mxx_prime = vmw.mxx;
        // myy_prime = vmw.myy;
        // mzz_prime = vmw.mzz;
        // myz_prime = vmw.myz;
    }

    if constexpr (MASS_CONSERV == MassBC ::Equilibrium)
    {
        numerical_solution_rhoeq_rotated(unit_nx, unit_ny, cylinder, nodeType, ux_prime, uy_prime, uz_prime,
                                         mxx_prime, myy_prime, mzz_prime, myz_prime,
                                         rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, NODE_TYPE, iter);
    }
    else if constexpr (MASS_CONSERV == MassBC ::Strong)
    {
        numerical_solution_strong_rotated(unit_nx, unit_ny, cylinder, nodeType, ux_prime, uy_prime, uz_prime,
                                          mxx_prime, myy_prime, mzz_prime, myz_prime,
                                          rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, NODE_TYPE, iter);
    }
}
