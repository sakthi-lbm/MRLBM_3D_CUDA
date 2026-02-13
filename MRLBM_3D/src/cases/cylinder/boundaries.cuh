#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../../nodeTypeMap.h"
#include "constants.h"
#include "../../globalStructs.h"

__host__ __device__ inline nodeType_t boundary_definitions(const int x, const int y, const int z)
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

    // --- CORNERS (3-way intersection) ---
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

    // --- EDGES (2-way intersection) ---
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

    // --- FACES (1-way intersection) ---
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
                                          real &rhoVar, real &ux, real &uy, real &uz,
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

        const real rho = toReal(6) * rho_I / toReal(5);

        rhoVar = rho;
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);
        mxx = toReal(0);
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

        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;

        const real rho = toReal(6) * rho_I / toReal(5);

        rhoVar = rho;
        ux = toReal(0);
        uy = toReal(0);
        uz = toReal(0);
        mxx = toReal(0);
        mxy = toReal(2) * mxy_I * rho_I / rho;
        mxz = toReal(2) * mxz_I * rho_I / rho;
        myy = toReal(0);
        myz = toReal(0);
        mzz = toReal(0);

        return;
    }
    case NORTH:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[7] + pop[19] + pop[21]) - (pop[14] + pop[24] + pop[25])) * inv_rho_I;
        const real myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        const real rho = toReal(6) * rho_I / toReal(5);

        rhoVar = rho;
        ux = U_MAX;                                                          // ux
        uy = toReal(0);                                                      // uy
        uz = toReal(0);                                                      // uz
        mxx = U_MAX * U_MAX;                                                 // mxx
        mxy = (toReal(6) * mxy_I * rho_I - U_MAX * rho) / (toReal(3) * rho); // mxy
        mxz = toReal(0);                                                     // mxz
        myy = toReal(0);                                                     // myy
        myz = toReal(2) * myz_I * rho_I / rho;                               // myz
        mzz = toReal(0);                                                     // mzz

        return;
    }
    case SOUTH:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = ((pop[8] + pop[20] + pop[22]) - (pop[13] + pop[23] + pop[26])) * inv_rho_I;
        const real myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        const real rho = toReal(6) * rho_I / toReal(5);

        rhoVar = rho;
        ux = toReal(0);                        // ux
        uy = toReal(0);                        // uy0
        uz = toReal(0);                        // uz
        mxx = toReal(0);                       // mxx
        mxy = toReal(2) * mxy_I * rho_I / rho; // mxy
        mxz = toReal(0);                       // mxz
        myy = toReal(0);                       // myy
        myz = toReal(2) * myz_I * rho_I / rho; // myz
        mzz = toReal(0);                       // mzz

        return;
    }
    case BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = ((pop[10] + pop[20] + pop[24]) - (pop[15] + pop[21] + pop[26])) * inv_rho_I;
        const real myz_I = ((pop[12] + pop[20] + pop[26]) - (pop[17] + pop[21] + pop[24])) * inv_rho_I;

        const real rho = toReal(6) * rho_I / toReal(5);

        rhoVar = rho;
        ux = toReal(0);                        // ux
        uy = toReal(0);                        // uy
        uz = toReal(0);                        // uz
        mxx = toReal(0);                       // mxx
        mxy = toReal(0);                       // mxy
        mxz = toReal(2) * mxz_I * rho_I / rho; // mxz
        myy = toReal(0);                       // myy
        myz = toReal(2) * myz_I * rho_I / rho; // myz
        mzz = toReal(0);                       // mzz

        return;
    }
    case FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = ((pop[9] + pop[19] + pop[23]) - (pop[16] + pop[22] + pop[25])) * inv_rho_I;
        const real myz_I = ((pop[11] + pop[19] + pop[25]) - (pop[18] + pop[22] + pop[23])) * inv_rho_I;

        const real rho = toReal(6) * rho_I / toReal(5);

        rhoVar = rho;
        ux = toReal(0);                        // ux
        uy = toReal(0);                        // uy
        uz = toReal(0);                        // uz
        mxx = toReal(0);                       // mxx
        mxy = toReal(0);                       // mxy
        mxz = toReal(2) * mxz_I * rho_I / rho; // mxz
        myy = toReal(0);                       // myy
        myz = toReal(2) * myz_I * rho_I / rho; // myz
        mzz = toReal(0);                       // mzz

        return;
    }
    case NORTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[24] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[14] + pop[24] + pop[25]) * inv_rho_I;

        const real rho = -toReal(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) /
                         (toReal(24) + toReal(18) * U_MAX - toReal(18) * U_MAX * U_MAX + OMEGA - toReal(3) * U_MAX * OMEGA + toReal(3) * U_MAX * U_MAX * OMEGA);

        rhoVar = rho;
        ux = U_MAX;          // ux
        uy = toReal(0);      // uy
        uz = toReal(0);      // uz
        mxx = U_MAX * U_MAX; // mxx
        mxy = (toReal(36) * mxy_I * rho_I + rho - toReal(3) * U_MAX * rho + toReal(3) * U_MAX * U_MAX * rho) /
              (toReal(9) * rho); // mxy
        mxz = toReal(0);         // mxz
        myy = toReal(0);         // myy
        myz = toReal(0);         // myz
        mzz = toReal(0);         // mzz

        return;
    }
    case SOUTH_WEST:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18] + pop[20] + pop[22];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[8] + pop[20] + pop[22]) * inv_rho_I;

        const real rho = toReal(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = (toReal(36) * mxy_I * rho_I - rho) / (toReal(9) * rho); // mxy
        mxz = toReal(0);                                              // mxz
        myy = toReal(0);                                              // myy
        myz = toReal(0);                                              // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18] + pop[22] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[16] + pop[22] + pop[25]) * inv_rho_I;

        const real rho = -toReal(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = toReal(0);                                              // mxy
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho); // mxz
        myy = toReal(0);                                              // myy
        myz = toReal(0);                                              // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17] + pop[20] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[10] + pop[20] + pop[24]) * inv_rho_I;

        const real rho = toReal(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = toReal(0);                                              // mxy
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho); // mxz
        myy = toReal(0);                                              // myy
        myz = toReal(0);                                              // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case NORTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17] + pop[19] + pop[21];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = (pop[7] + pop[19] + pop[21]) * inv_rho_I;

        const real rho = toReal(36) * (rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) /
                         (toReal(24) - toReal(18) * U_MAX - toReal(18) * U_MAX * U_MAX + OMEGA + toReal(3) * U_MAX * OMEGA + toReal(3) * U_MAX * U_MAX * OMEGA);

        rhoVar = rho;
        ux = U_MAX;          // ux
        uy = toReal(0);      // uy
        uz = toReal(0);      // uz
        mxx = U_MAX * U_MAX; // mxx
        mxy = (toReal(36) * mxy_I * rho_I - rho - toReal(3) * U_MAX * rho - toReal(3) * U_MAX * U_MAX * rho) /
              (toReal(9) * rho); // mxy
        mxz = toReal(0);         // mxz
        myy = toReal(0);         // myy
        myz = toReal(0);         // myz
        mzz = toReal(0);         // mzz

        return;
    }
    case SOUTH_EAST:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18] + pop[23] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxy_I = -(pop[13] + pop[23] + pop[26]) * inv_rho_I;

        const real rho = -toReal(36) * (-rho_I - mxy_I * rho_I + mxy_I * rho_I * OMEGA) / (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = (toReal(36) * mxy_I * rho_I + rho) / (toReal(9) * rho); // mxy
        mxz = toReal(0);                                              // mxz
        myy = toReal(0);                                              // myy
        myz = toReal(0);                                              // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18] + pop[19] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = (pop[9] + pop[19] + pop[23]) * inv_rho_I;

        const real rho = toReal(36) * (rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = toReal(0);                                              // mxy
        mxz = (toReal(36) * mxz_I * rho_I - rho) / (toReal(9) * rho); // mxz
        myy = toReal(0);                                              // myy
        myz = toReal(0);                                              // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17] + pop[21] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real mxz_I = -(pop[15] + pop[21] + pop[26]) * inv_rho_I;

        const real rho = -toReal(36) * (-rho_I - mxz_I * rho_I + mxz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = toReal(0);                                              // mxy
        mxz = (toReal(36) * mxz_I * rho_I + rho) / (toReal(9) * rho); // mxz
        myy = toReal(0);                                              // myy
        myz = toReal(0);                                              // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case NORTH_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16] + pop[19] + pop[25];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = (pop[11] + pop[19] + pop[25]) * inv_rho_I;

        const real rho = toReal(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = U_MAX;                                                   // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = U_MAX * U_MAX;                                          // mxx
        mxy = toReal(0);                                              // mxy
        mxz = toReal(0);                                              // mxz
        myy = toReal(0);                                              // myy
        myz = (toReal(36) * myz_I * rho_I - rho) / (toReal(9) * rho); // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case NORTH_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17] + pop[21] + pop[24];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = -(pop[17] + pop[21] + pop[24]) * inv_rho_I;

        const real rho = -toReal(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = U_MAX;                                                   // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = U_MAX * U_MAX;                                          // mxx
        mxy = toReal(0);                                              // mxy
        mxz = toReal(0);                                              // mxz
        myy = toReal(0);                                              // myy
        myz = (toReal(36) * myz_I * rho_I + rho) / (toReal(9) * rho); // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case SOUTH_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18] + pop[22] + pop[23];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = -(pop[18] + pop[22] + pop[23]) * inv_rho_I;

        const real rho = -toReal(36) * (-rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = toReal(0);                                              // mxy
        mxz = toReal(0);                                              // mxz
        myy = toReal(0);                                              // myy
        myz = (toReal(36) * myz_I * rho_I + rho) / (toReal(9) * rho); // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case SOUTH_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[20] + pop[26];
        const real inv_rho_I = toReal(1) / rho_I;

        const real myz_I = (pop[12] + pop[20] + pop[26]) * inv_rho_I;

        const real rho = toReal(36) * (rho_I - myz_I * rho_I + myz_I * rho_I * OMEGA) /
                         (toReal(24) + OMEGA);

        rhoVar = rho;
        ux = toReal(0);                                               // ux
        uy = toReal(0);                                               // uy
        uz = toReal(0);                                               // uz
        mxx = toReal(0);                                              // mxx
        mxy = toReal(0);                                              // mxy
        mxz = toReal(0);                                              // mxz
        myy = toReal(0);                                              // myy
        myz = (toReal(36) * myz_I * rho_I - rho) / (toReal(9) * rho); // myz
        mzz = toReal(0);                                              // mzz

        return;
    }
    case NORTH_WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16] + pop[25];

        const real rho = -toReal(216) * rho_I /
                         (-toReal(125) - toReal(75) * U_MAX + toReal(75) * U_MAX * U_MAX);

        rhoVar = rho;
        ux = U_MAX;          // ux
        uy = toReal(0);      // uy
        uz = toReal(0);      // uz
        mxx = U_MAX * U_MAX; // mxx
        mxy = toReal(0);     // mxy
        mxz = toReal(0);     // mxz
        myy = toReal(0);     // myy
        myz = toReal(0);     // myz
        mzz = toReal(0);     // mzz

        return;
    }
    case NORTH_WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17] + pop[24];

        const real rho = -toReal(216) * rho_I /
                         (-toReal(125) - toReal(75) * U_MAX + toReal(75) * U_MAX * U_MAX);

        rhoVar = rho;
        ux = U_MAX;          // ux
        uy = toReal(0);      // uy
        uz = toReal(0);      // uz
        mxx = U_MAX * U_MAX; // mxx
        mxy = toReal(0);     // mxy
        mxz = toReal(0);     // mxz
        myy = toReal(0);     // myy
        myz = toReal(0);     // myz
        mzz = toReal(0);     // mzz

        return;
    }
    case SOUTH_WEST_FRONT:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18] + pop[22];

        const real rho = toReal(216) * rho_I / toReal(125);

        rhoVar = rho;
        ux = toReal(0);  // ux
        uy = toReal(0);  // uy
        uz = toReal(0);  // uz
        mxx = toReal(0); // mxx
        mxy = toReal(0); // mxy
        mxz = toReal(0); // mxz
        myy = toReal(0); // myy
        myz = toReal(0); // myz
        mzz = toReal(0); // mzz

        return;
    }
    case SOUTH_WEST_BACK:
    {
        const real rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[20];

        const real rho = toReal(216) * rho_I / toReal(125);

        rhoVar = rho;
        ux = toReal(0);  // ux
        uy = toReal(0);  // uy
        uz = toReal(0);  // uz
        mxx = toReal(0); // mxx
        mxy = toReal(0); // mxy
        mxz = toReal(0); // mxz
        myy = toReal(0); // myy
        myz = toReal(0); // myz
        mzz = toReal(0); // mzz

        return;
    }
    case NORTH_EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[19];

        const real rho = -toReal(216) * rho_I /
                         (-toReal(125) + toReal(75) * U_MAX + toReal(75) * U_MAX * U_MAX);

        rhoVar = rho;
        ux = U_MAX;          // ux
        uy = toReal(0);      // uy
        uz = toReal(0);      // uz
        mxx = U_MAX * U_MAX; // mxx
        mxy = toReal(0);     // mxy
        mxz = toReal(0);     // mxz
        myy = toReal(0);     // myy
        myz = toReal(0);     // myz
        mzz = toReal(0);     // mzz

        return;
    }
    case NORTH_EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17] + pop[21];

        const real rho = -toReal(216) * rho_I /
                         (-toReal(125) + toReal(75) * U_MAX + toReal(75) * U_MAX * U_MAX);

        rhoVar = rho;
        ux = U_MAX;          // ux
        uy = toReal(0);      // uy
        uz = toReal(0);      // uz
        mxx = U_MAX * U_MAX; // mxx
        mxy = toReal(0);     // mxy
        mxz = toReal(0);     // mxz
        myy = toReal(0);     // myy
        myz = toReal(0);     // myz
        mzz = toReal(0);     // mzz

        return;
    }
    case SOUTH_EAST_FRONT:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18] + pop[23];

        const real rho = toReal(216) * rho_I / toReal(125);

        rhoVar = rho;
        ux = toReal(0);  // ux
        uy = toReal(0);  // uy
        uz = toReal(0);  // uz
        mxx = toReal(0); // mxx
        mxy = toReal(0); // mxy
        mxz = toReal(0); // mxz
        myy = toReal(0); // myy
        myz = toReal(0); // myz
        mzz = toReal(0); // mzz

        return;
    }
    case SOUTH_EAST_BACK:
    {
        const real rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15] + pop[26];

        const real rho = toReal(216) * rho_I / toReal(125);

        rhoVar = rho;
        ux = toReal(0);  // ux
        uy = toReal(0);  // uy
        uz = toReal(0);  // uz
        mxx = toReal(0); // mxx
        mxy = toReal(0); // mxy
        mxz = toReal(0); // mxz
        myy = toReal(0); // myy
        myz = toReal(0); // myz
        mzz = toReal(0); // mzz

        return;
    }
    }
}

#endif // BOUDARIES_H