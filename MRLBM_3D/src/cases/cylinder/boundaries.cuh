#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../../nodeTypeMap.h"
#include "constants.h"
#include "../../globalStructs.h"
#include "../../index.h"

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

__device__ inline void boundary_condition(nodeType_t nodeType, nodeVar fMom, real *pop,
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
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myy = toReal(0);
        myz = toReal(0);
        mzz = toReal(0);

        return;
    }
    case EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxx_I = (pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real myy_I = (pop[3] + pop[4] + pop[7] + pop[11] + pop[12] + pop[13] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real mzz_I = (pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[15] + pop[17] + pop[18] + pop[19] + pop[21] + pop[23] + pop[26]) * inv_rho_I - cs2;
        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const real myz_I = (pop[11] + pop[12] - pop[17] - pop[18] + pop[19] - pop[21] - pop[23] + pop[26]) * inv_rho_I;

        const size_t idx = IDX_BLOCK(threadIdx.x - 1, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
        rho = RHO_0 + fMom.rho[idx];
        // rho = -toReal(6) * rho_I / (-toReal(5) + toReal(3) * ux + toReal(3) * ux * ux);
        ux = fMom.ux[idx];
        uy = fMom.uy[idx];
        uz = fMom.uz[idx];

        mxx = ux * ux;
        // mxx = (toReal(9) * mxx_I * rho_I + rho - toReal(3) * ux * rho) / (toReal(6) * rho);
        myy = toReal(6) * myy_I * rho_I / (toReal(5) * rho);
        mzz = toReal(6) * mzz_I * rho_I / (toReal(5) * rho);
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

        rho = toReal(36) * rho_I * (toReal(1) + mxy_I - mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[8] + pop[20] + pop[22]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[16] + pop[22] + pop[25]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxz_I - mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho);
        myz = toReal(0);

        return;
    }
    case WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[10] + pop[20] + pop[24]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxz_I + mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho);
        myz = toReal(0);

        return;
    }
    case NORTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[7] + pop[19] + pop[21]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case SOUTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[13] + pop[23] + pop[26]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxy_I + mxy_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho);
        mxz = toReal(0);
        myz = toReal(0);

        return;
    }
    case EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[9] + pop[19] + pop[23]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) - mxz_I + mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho);
        myz = toReal(0);

        return;
    }
    case EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[15] + pop[21] + pop[26]) * inv_rho_I;

        rho = toReal(36) * rho_I * (toReal(1) + mxz_I - mxz_I * OMEGA) / (toReal(24) + OMEGA);
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);

        mxx = toReal(0);
        myy = toReal(0);
        mzz = toReal(0);
        mxy = toReal(0);
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho);
        myz = toReal(0);

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
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];

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
    case NORTH_EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];

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
    case SOUTH_EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];

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
    case SOUTH_EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];

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
    }
}

#endif // BOUDARIES_H