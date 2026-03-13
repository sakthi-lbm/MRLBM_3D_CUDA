#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../../nodeTypeMap.h"
#include "constants.h"
#include "../../globalStructs.h"

__host__ __device__ inline nodeType_t boundary_definitions(const unsigned int x, const unsigned int y, const unsigned int z)
{
    // Determine boundary flags based on position
    bool isW = (x == 0);
    bool isE = (x == NX - 1);
    bool isS = (y == 0);
    bool isN = (y == NY - 1);
    bool isB = (z == 0);
    bool isF = (z == NZ - 1);

// Apply Periodicity: if a direction is periodic, it's treated as BULK (no boundary)
#if X_PERIODIC
    isW = false;
    isE = false;
#endif
#if Y_PERIODIC
    isS = false;
    isN = false;
#endif
#if Z_PERIODIC
    isB = false;
    isF = false;
#endif

    // --- CORNERS
    if (isN && isW && isF)
        return NORTH_WEST_FRONT;
    if (isN && isW && isB)
        return NORTH_WEST_BACK;
    if (isN && isE && isF)
        return NORTH_EAST_FRONT;
    if (isN && isE && isB)
        return NORTH_EAST_BACK;
    if (isS && isW && isF)
        return SOUTH_WEST_FRONT;
    if (isS && isW && isB)
        return SOUTH_WEST_BACK;
    if (isS && isE && isF)
        return SOUTH_EAST_FRONT;
    if (isS && isE && isB)
        return SOUTH_EAST_BACK;

    // --- EDGES
    if (isN && isW)
        return NORTH_WEST;
    if (isN && isE)
        return NORTH_EAST;
    if (isN && isF)
        return NORTH_FRONT;
    if (isN && isB)
        return NORTH_BACK;

    if (isS && isW)
        return SOUTH_WEST;
    if (isS && isE)
        return SOUTH_EAST;
    if (isS && isF)
        return SOUTH_FRONT;
    if (isS && isB)
        return SOUTH_BACK;

    if (isW && isF)
        return WEST_FRONT;
    if (isW && isB)
        return WEST_BACK;
    if (isE && isF)
        return EAST_FRONT;
    if (isE && isB)
        return EAST_BACK;

    // --- FACES
    if (isN)
        return NORTH;
    if (isS)
        return SOUTH;
    if (isW)
        return WEST;
    if (isE)
        return EAST;
    if (isF)
        return FRONT;
    if (isB)
        return BACK;

    // Default
    return BULK;
}

__device__ inline void current_update_neumaan_density_velocity(const real *s_pop,
                                                               real &rho, real &ux, real &uy, real &uz)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;
    const unsigned int xm1 = (tx + BLOCK_THREAD_X - 1) % BLOCK_THREAD_X;

    real rho_i = 0.0;
    real jx_i = 0.0;
    real jy_i = 0.0;
    real jz_i = 0.0;

    real rho_b = 0.0;
    real jx_b = 0.0;
    real jy_b = 0.0;
    real jz_b = 0.0;

#pragma unroll
    for (int q = 0; q < Q; q++)
    {
        real fi = s_pop[idxPopBlock(xm1, ty, tz, q)];

        rho_i += fi;
        jx_i += fi * toReal(d_cx[q]);
        jy_i += fi * toReal(d_cy[q]);
        jz_i += fi * toReal(d_cz[q]);

        if constexpr (CONVECTIVE_OUTLET)
        {
            real fb = s_pop[idxPopBlock(tx, ty, tz, q)];

            rho_b += fb;
            jx_b += fb * toReal(d_cx[q]);
            jy_b += fb * toReal(d_cy[q]);
            jz_b += fb * toReal(d_cz[q]);
        }
    }
    const real inv_rhoi = toReal(1) / rho_i;
    jx_i *= inv_rhoi;
    jy_i *= inv_rhoi;
    jz_i *= inv_rhoi;

    if constexpr (CONVECTIVE_OUTLET)
    {
        const real UC = d_UCONV;

        const real inv_rhob = toReal(1) / rho_b;
        jx_b *= inv_rhob;
        jy_b *= inv_rhob;
        jz_b *= inv_rhob;

        rho = (1.0 - UC) * rho_b + UC * rho_i;
        // rho = RHO_0;
        ux = (1.0 - UC) * jx_b + UC * jx_i;
        uy = (1.0 - UC) * jy_b + UC * jy_i;
        uz = (1.0 - UC) * jz_b + UC * jz_i;
    }
    else
    {
        rho = rho_i;
        // rho = RHO_0;
        ux = jx_i;
        uy = jy_i;
        uz = jz_i;
    }
}

__device__ inline void update_neumaan_density_velocity(nodeVar dMom, real &rho, real &ux, real &uy, real &uz)
{
    size_t idx;

    // interior node values
    idx = IDX_BLOCK(threadIdx.x - 1, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    const real rhoi = RHO_0 + dMom.rho[idx];
    const real uxi = dMom.ux[idx];
    const real uyi = dMom.uy[idx];
    const real uzi = dMom.uz[idx];

    if constexpr (CONVECTIVE_OUTLET)
    {
        // boundary node values
        idx = IDX_BLOCK(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        const real rhob = RHO_0 + dMom.rho[idx];
        const real uxb = dMom.ux[idx];
        const real uyb = dMom.uy[idx];
        const real uzb = dMom.uz[idx];

        // const real Uc = sqrt(uxi * uxi);
        const real Uc = d_UCONV;
        rho = (1.0 - Uc) * rhob + Uc * rhoi;
        // rho = RHO_0;
        ux = (1.0 - Uc) * uxb + Uc * uxi;
        uy = (1.0 - Uc) * uyb + Uc * uyi;
        uz = (1.0 - Uc) * uzb + Uc * uzi;
    }
    else
    {
        rho = rhoi;
        // rho = RHO_0;
        ux = uxi;
        uy = uyi;
        uz = uzi;
    }
}

__device__ inline void boundary_condition(nodeType_t nodeType, nodeVar dMom, real *pop, real *s_pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz)
{
    switch (nodeType)
    {
    case WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18] + pop[20] + pop[22] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const real mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;

        rho = -toReal(6) * rho_I / (-toReal(5) + toReal(3) * ux + toReal(3) * ux * ux);
        ux = U_MAX;
        uy = toReal(0);
        uz = toReal(0);

        mxx = ux * ux;
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myz = toReal(0);

        return;
    }
    case EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        // const real mxx_I = (pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real myy_I = (pop[3] + pop[4] + pop[7] + pop[11] + pop[12] + pop[13] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real mzz_I = (pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const real myz_I = (pop[11] + pop[12] - pop[17] - pop[18] + pop[19] - pop[21] - pop[23] + pop[26]) * inv_rho_I;

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        // rho = toReal(6) * rho_I / (toReal(5) + toReal(3) * ux + toReal(3) * ux * ux);

        // mxx = ux * ux;
        // // mxx = (toReal(9) * mxx_I * rho_I + rho - toReal(3) * ux * rho) / (toReal(6) * rho);
        // myy = toReal(6) * myy_I * rho_I / (toReal(5) * rho);
        // mzz = toReal(6) * mzz_I * rho_I / (toReal(5) * rho);
        // mxy = (toReal(6) * mxy_I * rho_I - uy * rho) / (toReal(3) * rho);
        // mxz = (toReal(6) * mxz_I * rho_I - uz * rho) / (toReal(3) * rho);
        // myz = toReal(6) * myz_I * rho_I / (toReal(5) * rho);

        mxx = ux * ux;
        myy = (toReal(6) * myy_I * rho_I - toReal(6) * mzz_I * rho_I +
               toReal(5) * uy * uy * rho + toReal(5) * uz * uz * rho) /
              (toReal(10) * rho);
        mzz = (-toReal(6) * myy_I * rho_I + toReal(6) * mzz_I * rho_I +
               toReal(5) * uy * uy * rho + toReal(5) * uz * uz * rho) /
              (toReal(10) * rho);
        mxy = (toReal(6) * mxy_I * rho_I - uy * rho) / (toReal(3) * rho);
        mxz = (toReal(6) * mxz_I * rho_I - uz * rho) / (toReal(3) * rho);
        myz = toReal(6) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case NORTH:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const real myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(0);
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case SOUTH:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(0);
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const real myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;
        const real myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        rho = toReal(6) * rho_I / toReal(5);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myz = toReal(2) * myz_I * rho_I / rho;

        return;
    }
    case NORTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[14] + pop[24] + pop[25]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[17] - pop[24] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + mxy_I - mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case SOUTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[8] + pop[20] + pop[22]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[18] + pop[20] - pop[22]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[16] + pop[22] + pop[25]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[18] - pop[22] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + mxz_I - mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[10] + pop[20] + pop[24]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[17] + pop[20] - pop[24]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxz_I + mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho);
        myz = toReal(12) * myz_I * rho_I / (toReal(5) * rho);

        return;
    }
    case NORTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[7] + pop[19] + pop[21]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[17] + pop[19] - pop[21]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) - mxy_I);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case SOUTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[13] + pop[23] + pop[26]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[18] - pop[23] + pop[26]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) + mxy_I - mxy_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) + mxy_I);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[9] + pop[19] + pop[23]) * inv_rho_I;
        const real myz_I = (pop[11] - pop[18] + pop[19] - pop[23]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) - mxz_I + mxz_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) - mxz_I);
        // rho = toReal(36) * rho_I /toReal(25);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[15] + pop[21] + pop[26]) * inv_rho_I;
        const real myz_I = (pop[12] - pop[17] - pop[21] + pop[26]) * inv_rho_I;

        // rho = toReal(36) * rho_I * (toReal(1) + mxz_I - mxz_I * OMEGA) / (toReal(24) + OMEGA);
        // rho = toReal(1.5) * rho_I * (toReal(1) + mxz_I);
        // rho = toReal(36) * rho_I /toReal(25);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho);
        myz = (toReal(12) * myz_I * rho_I) / (toReal(5) * rho);

        return;
    }
    case NORTH_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16] + pop[19] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = (pop[11] + pop[19] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - myz_I + myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I - rho) / (toReal(9) * rho);

        return;
    }
    case NORTH_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = -(pop[17] + pop[21] + pop[24]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + myz_I - myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I + rho) / (toReal(9) * rho);

        return;
    }
    case SOUTH_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18] + pop[22] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = -(pop[18] + pop[22] + pop[23]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + myz_I - myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I + rho) / (toReal(9) * rho);

        return;
    }
    case SOUTH_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[20] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = (pop[12] + pop[20] + pop[26]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - myz_I + myz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = (toReal(36) * myz_I * rho_I - rho) / (toReal(9) * rho);

        return;
    }
    case NORTH_WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16] + pop[25];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case NORTH_WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17] + pop[24];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18] + pop[22];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[20];

        rho = toReal(216) * rho_I / toReal(125);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case NORTH_EAST_FRONT:
    {
        // const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case NORTH_EAST_BACK:
    {
        // const real rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_EAST_FRONT:
    {
        // const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_EAST_BACK:
    {
        // const real rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];

        // rho = toReal(216) * rho_I / toReal(125);
        // ux = toReal(0);
        // uy = toReal(0);
        // uz = toReal(0);

        if constexpr (NEUMANN_CURRENT_UPDATE)
        {
            current_update_neumaan_density_velocity(s_pop, rho, ux, uy, uz);
        }
        else
        {
            update_neumaan_density_velocity(dMom, rho, ux, uy, uz);
        }

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    }
}

__device__ inline void fluid_boundary_condition(const nodeType_t nodeTag, const cylinderVar &cylinder,
                                                const real *pop, real &rho, real &ux, real &uy, real &uz,
                                                real &mxx, real &myy, real &mzz,
                                                real &mxy, real &mxz, real &myz)
{
    real rhoI = toReal(0.0);
    real uxI = toReal(0.0);
    real uyI = toReal(0.0);
    real uzI = toReal(0.0);
    real mxxI = toReal(0.0);
    real myyI = toReal(0.0);
    real mzzI = toReal(0.0);
    real mxyI = toReal(0.0);
    real mxzI = toReal(0.0);
    real myzI = toReal(0.0);

    uint32_t incoming_mask = cylinder.incomingMask_bcfluid[nodeTag];
    while (incoming_mask)
    {
        const int q = __ffs(incoming_mask) - 1;
        incoming_mask &= incoming_mask - 1;

        const real cx = toReal(d_cx[q]);
        const real cy = toReal(d_cy[q]);
        const real cz = toReal(d_cz[q]);

        const real Hxx = cx * cx - cs2;
        const real Hyy = cy * cy - cs2;
        const real Hzz = cz * cz - cs2;
        const real Hxy = cx * cy;
        const real Hxz = cx * cz;
        const real Hyz = cy * cz;

        const real fq = pop[q];
        rhoI += fq;
        uxI += fq * cx;
        uyI += fq * cy;
        uzI += fq * cz;
        mxxI += fq * Hxx;
        myyI += fq * Hyy;
        mzzI += fq * Hzz;
        mxyI += fq * Hxy;
        mxzI += fq * Hxz;
        myzI += fq * Hyz;
    }

    switch (nodeTag)
    {
    case 17:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = -9996 - 206 * OMEGA;
            const real b = -6 * rhoI * (12 * mxxI * (-17 + 20 * OMEGA) - 17 * (102 + 36 * mxyI + 12 * myyI - 5 * uxI - 5 * uyI) + OMEGA * (720 * mxyI + 240 * myyI + 53 * (uxI + uyI)));
            const real c = 3 * OMEGA * rhoI * rhoI * (45 * mxxI * mxxI + 405 * mxyI * mxyI + 45 * myyI * myyI + 345 * myyI * uxI + 589 * uxI * uxI + 345 * myyI * uyI + 1467 * uxI * uyI + 589 * uyI * uyI + 45 * mxyI * (6 * myyI + 23 * (uxI + uyI)) + 15 * mxxI * (18 * mxyI + 6 * myyI + 23 * (uxI + uyI)));

            const real disc = b * b - 4.0 * a * c;
            const real rho1 = (-b + sqrt(disc)) / (2.0 * a);
            const real rho2 = (-b - sqrt(disc)) / (2.0 * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
        }
        ux = (3 * mxxI * rhoI + 9 * mxyI * rhoI + 3 * myyI * rhoI + 20 * rhoI * uxI + 3 * rhoI * uyI + rho) / (17 * rho);
        uy = (3 * mxxI * rhoI + 9 * mxyI * rhoI + 3 * myyI * rhoI + 3 * rhoI * uxI + 20 * rhoI * uyI + rho) / (17 * rho);
        uz = (3 * rhoI * (mxzI + myzI + 10 * uzI)) / (29 * rho);
        mxx = (57 * mxxI * rhoI + 18 * mxyI * rhoI + 6 * myyI * rhoI + 6 * rhoI * uxI + 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        myy = (6 * mxxI * rhoI + 18 * mxyI * rhoI + 57 * myyI * rhoI + 6 * rhoI * uxI + 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        mzz = (36 * mzzI * rhoI) / (35 * rho);
        mxy = (3 * mxxI * rhoI + 26 * mxyI * rhoI + 3 * myyI * rhoI + 3 * rhoI * uxI + 3 * rhoI * uyI + rho) / (17 * rho);
        mxz = (rhoI * (32 * mxzI + 3 * myzI + uzI)) / (29 * rho);
        myz = (rhoI * (3 * mxzI + 32 * myzI + uzI)) / (29 * rho);

        break;
    }
    case 34:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = 9996 + 206 * OMEGA;
            const real b = 6 * rhoI * (mxyI * (612 - 720 * OMEGA) + 240 * myyI * OMEGA + 12 * mxxI * (-17 + 20 * OMEGA) - 17 * (102 + 12 * myyI + 5 * uxI - 5 * uyI) + 53 * OMEGA * (-uxI + uyI));
            const real c = -3 * OMEGA * rhoI * rhoI * (45 * mxxI * mxxI + 405 * mxyI * mxyI + 45 * myyI * myyI - 345 * myyI * uxI + 589 * uxI * uxI + 345 * myyI * uyI - 1467 * uxI * uyI + 589 * uyI * uyI - 45 * mxyI * (6 * myyI - 23 * uxI + 23 * uyI) + 15 * mxxI * (-18 * mxyI + 6 * myyI - 23 * uxI + 23 * uyI));

            const real disc = b * b - 4.0 * a * c;
            const real rho1 = (-b + sqrt(disc)) / (2.0 * a);
            const real rho2 = (-b - sqrt(disc)) / (2.0 * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
        }
        ux = -(3 * mxxI * rhoI - 9 * mxyI * rhoI + 3 * myyI * rhoI - 20 * rhoI * uxI + 3 * rhoI * uyI + rho) / (17 * rho);
        uy = (3 * mxxI * rhoI - 9 * mxyI * rhoI + 3 * myyI * rhoI - 3 * rhoI * uxI + 20 * rhoI * uyI + rho) / (17 * rho);
        uz = (3 * rhoI * (-mxzI + myzI + 10 * uzI)) / (29 * rho);
        mxx = (57 * mxxI * rhoI - 18 * mxyI * rhoI + 6 * myyI * rhoI - 6 * rhoI * uxI + 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        myy = (6 * mxxI * rhoI - 18 * mxyI * rhoI + 57 * myyI * rhoI - 6 * rhoI * uxI + 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        mzz = (36 * mzzI * rhoI) / (35 * rho);
        mxy = -(3 * mxxI * rhoI - 26 * mxyI * rhoI + 3 * myyI * rhoI - 3 * rhoI * uxI + 3 * rhoI * uyI + rho) / (17 * rho);
        mxz = (rhoI * (32 * mxzI - 3 * myzI - uzI)) / (29 * rho);
        myz = (rhoI * (-3 * mxzI + 32 * myzI + uzI)) / (29 * rho);

        break;
    }
    case 68:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = 9996 + 206 * OMEGA;
            const real b = 6 * rhoI * (mxyI * (612 - 720 * OMEGA) + 240 * myyI * OMEGA + 12 * mxxI * (-17 + 20 * OMEGA) + 53 * OMEGA * (uxI - uyI) - 17 * (102 + 12 * myyI - 5 * uxI + 5 * uyI));
            const real c = -3 * OMEGA * rhoI * rhoI * (45 * mxxI * mxxI + 405 * mxyI * mxyI + 45 * myyI * myyI + 345 * myyI * uxI + 589 * uxI * uxI - 45 * mxyI * (6 * myyI + 23 * uxI - 23 * uyI) + 15 * mxxI * (-18 * mxyI + 6 * myyI + 23 * uxI - 23 * uyI) - 345 * myyI * uyI - 1467 * uxI * uyI + 589 * uyI * uyI);

            const real disc = b * b - 4.0 * a * c;
            const real rho1 = (-b + sqrt(disc)) / (2.0 * a);
            const real rho2 = (-b - sqrt(disc)) / (2.0 * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
        }
        ux = (3 * mxxI * rhoI - 9 * mxyI * rhoI + 3 * myyI * rhoI + 20 * rhoI * uxI - 3 * rhoI * uyI + rho) / (17 * rho);
        uy = -(3 * mxxI * rhoI - 9 * mxyI * rhoI + 3 * myyI * rhoI + 3 * rhoI * uxI - 20 * rhoI * uyI + rho) / (17 * rho);
        uz = (3 * rhoI * (mxzI - myzI + 10 * uzI)) / (29 * rho);
        mxx = (57 * mxxI * rhoI - 18 * mxyI * rhoI + 6 * myyI * rhoI + 6 * rhoI * uxI - 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        myy = (6 * mxxI * rhoI - 18 * mxyI * rhoI + 57 * myyI * rhoI + 6 * rhoI * uxI - 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        mzz = (36 * mzzI * rhoI) / (35 * rho);
        mxy = -(3 * mxxI * rhoI - 26 * mxyI * rhoI + 3 * myyI * rhoI + 3 * rhoI * uxI - 3 * rhoI * uyI + rho) / (17 * rho);
        mxz = (rhoI * (32 * mxzI - 3 * myzI + uzI)) / (29 * rho);
        myz = -(rhoI * (3 * mxzI - 32 * myzI + uzI)) / (29 * rho);

        break;
    }
    case 136:
    {
        if constexpr (BCF_MASS_CONSERV == MassBC::Strong)
        {
            const real a = -9996 - 206 * OMEGA;
            const real b = -6 * rhoI * (12 * mxxI * (-17 + 20 * OMEGA) - 17 * (102 + 36 * mxyI + 12 * myyI + 5 * uxI + 5 * uyI) + OMEGA * (720 * mxyI + 240 * myyI - 53 * (uxI + uyI)));
            const real c = 3 * OMEGA * rhoI * rhoI * (45 * mxxI * mxxI + 405 * mxyI * mxyI + 45 * myyI * myyI - 345 * myyI * uxI + 589 * uxI * uxI - 345 * myyI * uyI + 1467 * uxI * uyI + 589 * uyI * uyI + 45 * mxyI * (6 * myyI - 23 * (uxI + uyI)) + 15 * mxxI * (18 * mxyI + 6 * myyI - 23 * (uxI + uyI)));

            const real disc = b * b - 4.0 * a * c;
            const real rho1 = (-b + sqrt(disc)) / (2.0 * a);
            const real rho2 = (-b - sqrt(disc)) / (2.0 * a);
            rho = (rho1 > 0.0 && fabs(rho1 - rhoI) < fabs(rho2 - rhoI)) ? rho1 : rho2;
        }
        else if constexpr (BCF_MASS_CONSERV == MassBC::Equilibrium)
        {
        }
        ux = -(3 * mxxI * rhoI + 9 * mxyI * rhoI + 3 * myyI * rhoI - 20 * rhoI * uxI - 3 * rhoI * uyI + rho) / (17 * rho);
        uy = -(3 * mxxI * rhoI + 9 * mxyI * rhoI + 3 * myyI * rhoI - 3 * rhoI * uxI - 20 * rhoI * uyI + rho) / (17 * rho);
        uz = (-3 * rhoI * (mxzI + myzI - 10 * uzI)) / (29 * rho);
        mxx = (57 * mxxI * rhoI + 18 * mxyI * rhoI + 6 * myyI * rhoI - 6 * rhoI * uxI - 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        myy = (6 * mxxI * rhoI + 18 * mxyI * rhoI + 57 * myyI * rhoI - 6 * rhoI * uxI - 6 * rhoI * uyI + 2 * rho) / (51 * rho);
        mzz = (36 * mzzI * rhoI) / (35 * rho);
        mxy = (3 * mxxI * rhoI + 26 * mxyI * rhoI + 3 * myyI * rhoI - 3 * rhoI * uxI - 3 * rhoI * uyI + rho) / (17 * rho);
        mxz = (rhoI * (32 * mxzI + 3 * myzI - uzI)) / (29 * rho);
        myz = (rhoI * (3 * mxzI + 32 * myzI - uzI)) / (29 * rho);

        break;
    }
    }
}

#endif // BOUDARIES_H