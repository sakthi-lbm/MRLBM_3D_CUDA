#ifndef HALO_INTERFACE_H
#define HALO_INTERFACE_H

__device__ inline void pop_load_from_halo(haloData fHalo, unsigned int tx, unsigned int ty, unsigned int tz,
                                          unsigned int bx, unsigned int by, unsigned int bz, real *pop)
{
    const unsigned int txm1 = (tx - 1 + blockDim.x) % blockDim.x;
    const unsigned int txp1 = (tx + 1 + blockDim.x) % blockDim.x;
    const unsigned int tym1 = (ty - 1 + blockDim.y) % blockDim.y;
    const unsigned int typ1 = (ty + 1 + blockDim.y) % blockDim.y;
    const unsigned int tzm1 = (tz - 1 + blockDim.z) % blockDim.z;
    const unsigned int tzp1 = (tz + 1 + blockDim.z) % blockDim.z;

    const unsigned int bxm1 = (bx - 1 + gridDim.x) % gridDim.x;
    const unsigned int bxp1 = (bx + 1 + gridDim.x) % gridDim.x;
    const unsigned int bym1 = (by - 1 + gridDim.y) % gridDim.y;
    const unsigned int byp1 = (by + 1 + gridDim.y) % gridDim.y;
    const unsigned int bzm1 = (bz - 1 + gridDim.z) % gridDim.z;
    const unsigned int bzp1 = (bz + 1 + gridDim.z) % gridDim.z;

    if (tx == 0)
    {
        // WEST face in the block
        pop[1] = fHalo.X_EAST[idxPopX(ty, tz, 0, bxm1, by, bz)];
        pop[7] = fHalo.X_EAST[idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by), bz)];
        pop[9] = fHalo.X_EAST[idxPopX(ty, tzm1, 2, bxm1, by, ((tz == 0) ? bzm1 : bz))];
        pop[13] = fHalo.X_EAST[idxPopX(typ1, tz, 3, bxm1, ((ty == (blockDim.y - 1)) ? byp1 : by), bz)];
        pop[15] = fHalo.X_EAST[idxPopX(ty, tzp1, 4, bxm1, by, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[19] = fHalo.X_EAST[idxPopX(tym1, tzm1, 5, bxm1, ((ty == 0) ? bym1 : by), ((tz == 0) ? bzm1 : bz))];
        pop[21] = fHalo.X_EAST[idxPopX(tym1, tzp1, 6, bxm1, ((ty == 0) ? bym1 : by), ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[23] = fHalo.X_EAST[idxPopX(typ1, tzm1, 7, bxm1, ((ty == (blockDim.y - 1)) ? byp1 : by), ((tz == 0) ? bzm1 : bz))];
        pop[26] = fHalo.X_EAST[idxPopX(typ1, tzp1, 8, bxm1, ((ty == (blockDim.y - 1)) ? byp1 : by), ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
    }

    if (tx == (blockDim.x - 1))
    {
        // EAST face in the block
        pop[2] = fHalo.X_WEST[idxPopX(ty, tz, 0, bxp1, by, bz)];
        pop[8] = fHalo.X_WEST[idxPopX(typ1, tz, 1, bxp1, ((ty == (blockDim.y - 1)) ? byp1 : by), bz)];
        pop[10] = fHalo.X_WEST[idxPopX(ty, tzp1, 2, bxp1, by, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[14] = fHalo.X_WEST[idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by), bz)];
        pop[16] = fHalo.X_WEST[idxPopX(ty, tzm1, 4, bxp1, by, ((tz == 0) ? bzm1 : bz))];
        pop[20] = fHalo.X_WEST[idxPopX(typ1, tzp1, 5, bxp1, ((ty == (blockDim.y - 1)) ? byp1 : by), ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[22] = fHalo.X_WEST[idxPopX(typ1, tzm1, 6, bxp1, ((ty == (blockDim.y - 1)) ? byp1 : by), ((tz == 0) ? bzm1 : bz))];
        pop[24] = fHalo.X_WEST[idxPopX(tym1, tzp1, 7, bxp1, ((ty == 0) ? bym1 : by), ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[25] = fHalo.X_WEST[idxPopX(tym1, tzm1, 8, bxp1, ((ty == 0) ? bym1 : by), ((tz == 0) ? bzm1 : bz))];

        // printf("BX=%u BY=%u | LOAD EAST[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[3], pop[6], pop[7]);
    }

    if (ty == 0)
    {
        // SOUTH face in the block
        pop[3] = fHalo.Y_NORTH[idxPopY(tx, tz, 0, bx, bym1, bz)];
        pop[7] = fHalo.Y_NORTH[idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx), bym1, bz)];
        pop[11] = fHalo.Y_NORTH[idxPopY(tx, tzm1, 2, bx, bym1, ((tz == 0) ? bzm1 : bz))];
        pop[14] = fHalo.Y_NORTH[idxPopY(txp1, tz, 3, ((tx == (blockDim.x - 1)) ? bxp1 : bx), bym1, bz)];
        pop[17] = fHalo.Y_NORTH[idxPopY(tx, tzp1, 4, bx, bym1, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[19] = fHalo.Y_NORTH[idxPopY(txm1, tzm1, 5, ((tx == 0) ? bxm1 : bx), bym1, ((tz == 0) ? bzm1 : bz))];
        pop[21] = fHalo.Y_NORTH[idxPopY(txm1, tzp1, 6, ((tx == 0) ? bxm1 : bx), bym1, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[24] = fHalo.Y_NORTH[idxPopY(txp1, tzp1, 7, ((tx == (blockDim.x - 1)) ? bxp1 : bx), bym1, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[25] = fHalo.Y_NORTH[idxPopY(txp1, tzm1, 8, ((tx == (blockDim.x - 1)) ? bxp1 : bx), bym1, ((tz == 0) ? bzm1 : bz))];
    }

    if (ty == (blockDim.y - 1))
    {
        // NORTH face in the block
        pop[4] = fHalo.Y_SOUTH[idxPopY(tx, tz, 0, bx, byp1, bz)];
        pop[8] = fHalo.Y_SOUTH[idxPopY(txp1, tz, 1, ((tx == (blockDim.x - 1)) ? bxp1 : bx), byp1, bz)];
        pop[12] = fHalo.Y_SOUTH[idxPopY(tx, tzp1, 2, bx, byp1, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[13] = fHalo.Y_SOUTH[idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx), byp1, bz)];
        pop[18] = fHalo.Y_SOUTH[idxPopY(tx, tzm1, 4, bx, byp1, ((tz == 0) ? bzm1 : bz))];
        pop[20] = fHalo.Y_SOUTH[idxPopY(txp1, tzp1, 5, ((tx == (blockDim.x - 1)) ? bxp1 : bx), byp1, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
        pop[22] = fHalo.Y_SOUTH[idxPopY(txp1, tzm1, 6, ((tx == (blockDim.x - 1)) ? bxp1 : bx), byp1, ((tz == 0) ? bzm1 : bz))];
        pop[23] = fHalo.Y_SOUTH[idxPopY(txm1, tzm1, 7, ((tx == 0) ? bxm1 : bx), byp1, ((tz == 0) ? bzm1 : bz))];
        pop[26] = fHalo.Y_SOUTH[idxPopY(txm1, tzp1, 8, ((tx == 0) ? bxm1 : bx), byp1, ((tz == (blockDim.z - 1)) ? bzp1 : bz))];
    }

    if (tz == 0)
    {
        // BACK face in the block
        pop[5] = fHalo.Z_FRONT[idxPopZ(tx, ty, 0, bx, by, bzm1)];
        pop[9] = fHalo.Z_FRONT[idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx), by, bzm1)];
        pop[11] = fHalo.Z_FRONT[idxPopZ(tx, tym1, 2, bx, ((ty == 0) ? bym1 : by), bzm1)];
        pop[16] = fHalo.Z_FRONT[idxPopZ(txp1, ty, 3, ((tx == (blockDim.x - 1)) ? bxp1 : bx), by, bzm1)];
        pop[18] = fHalo.Z_FRONT[idxPopZ(tx, typ1, 4, bx, ((ty == (blockDim.y - 1)) ? byp1 : by), bzm1)];
        pop[19] = fHalo.Z_FRONT[idxPopZ(txm1, tym1, 5, ((tx == 0) ? bxm1 : bx), ((ty == 0) ? bym1 : by), bzm1)];
        pop[22] = fHalo.Z_FRONT[idxPopZ(txp1, typ1, 6, ((tx == (blockDim.x - 1)) ? bxp1 : bx), ((ty == (blockDim.y - 1)) ? byp1 : by), bzm1)];
        pop[23] = fHalo.Z_FRONT[idxPopZ(txm1, typ1, 7, ((tx == 0) ? bxm1 : bx), ((ty == (blockDim.y - 1)) ? byp1 : by), bzm1)];
        pop[25] = fHalo.Z_FRONT[idxPopZ(txp1, tym1, 8, ((tx == (blockDim.x - 1)) ? bxp1 : bx), ((ty == 0) ? bym1 : by), bzm1)];
    }

    if (tz == (blockDim.z - 1))
    {
        // FRONT face in the block
        pop[6] = fHalo.Z_BACK[idxPopZ(tx, ty, 0, bx, by, bzp1)];
        pop[10] = fHalo.Z_BACK[idxPopZ(txp1, ty, 1, ((tx == (blockDim.x - 1)) ? bxp1 : bx), by, bzp1)];
        pop[12] = fHalo.Z_BACK[idxPopZ(tx, typ1, 2, bx, ((ty == (blockDim.y - 1)) ? byp1 : by), bzp1)];
        pop[15] = fHalo.Z_BACK[idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx), by, bzp1)];
        pop[17] = fHalo.Z_BACK[idxPopZ(tx, tym1, 4, bx, ((ty == 0) ? bym1 : by), bzp1)];
        pop[20] = fHalo.Z_BACK[idxPopZ(txp1, typ1, 5, ((tx == (blockDim.x - 1)) ? bxp1 : bx), ((ty == (blockDim.y - 1)) ? byp1 : by), bzp1)];
        pop[21] = fHalo.Z_BACK[idxPopZ(txm1, tym1, 6, ((tx == 0) ? bxm1 : bx), ((ty == 0) ? bym1 : by), bzp1)];
        pop[24] = fHalo.Z_BACK[idxPopZ(txp1, tym1, 7, ((tx == (blockDim.x - 1)) ? bxp1 : bx), ((ty == 0) ? bym1 : by), bzp1)];
        pop[26] = fHalo.Z_BACK[idxPopZ(txm1, typ1, 8, ((tx == 0) ? bxm1 : bx), ((ty == (blockDim.y - 1)) ? byp1 : by), bzp1)];
    }
}

__device__ inline void pop_save_to_halo(haloData gHalo, unsigned int tx, unsigned int ty, unsigned int tz,
                                        unsigned int bx, unsigned int by, unsigned int bz, real *pop)
{
    if (tx == 0)
    {
        // WEST FACE
        gHalo.X_WEST[idxPopX(ty, tz, 0, bx, by, bz)] = pop[2];
        gHalo.X_WEST[idxPopX(ty, tz, 1, bx, by, bz)] = pop[8];
        gHalo.X_WEST[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
        gHalo.X_WEST[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
        gHalo.X_WEST[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
        gHalo.X_WEST[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
        gHalo.X_WEST[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
        gHalo.X_WEST[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
        gHalo.X_WEST[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
        // printf("BX=%u BY=%u | WEST[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[3], pop[6], pop[7]);
    }

    if (tx == (blockDim.x - 1))
    {
        // EAST FACE
        gHalo.X_EAST[idxPopX(ty, tz, 0, bx, by, bz)] = pop[1];
        gHalo.X_EAST[idxPopX(ty, tz, 1, bx, by, bz)] = pop[7];
        gHalo.X_EAST[idxPopX(ty, tz, 2, bx, by, bz)] = pop[9];
        gHalo.X_EAST[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
        gHalo.X_EAST[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
        gHalo.X_EAST[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
        gHalo.X_EAST[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
        gHalo.X_EAST[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
        gHalo.X_EAST[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
        // printf("BX=%u BY=%u | EAST[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[1], pop[5], pop[8]);
    }

    if (ty == 0)
    {
        // SOUTH FACE
        gHalo.Y_SOUTH[idxPopY(tx, tz, 0, bx, by, bz)] = pop[4];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 1, bx, by, bz)] = pop[8];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 2, bx, by, bz)] = pop[12];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 3, bx, by, bz)] = pop[13];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 4, bx, by, bz)] = pop[18];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
        gHalo.Y_SOUTH[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
        // printf("BX=%u BY=%u | SOUTH[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[4], pop[7], pop[8]);
    }

    if (ty == (blockDim.y - 1))
    {
        // NORTH FACE
        gHalo.Y_NORTH[idxPopY(tx, tz, 0, bx, by, bz)] = pop[3];
        gHalo.Y_NORTH[idxPopY(tx, tz, 1, bx, by, bz)] = pop[7];
        gHalo.Y_NORTH[idxPopY(tx, tz, 2, bx, by, bz)] = pop[11];
        gHalo.Y_NORTH[idxPopY(tx, tz, 3, bx, by, bz)] = pop[14];
        gHalo.Y_NORTH[idxPopY(tx, tz, 4, bx, by, bz)] = pop[17];
        gHalo.Y_NORTH[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
        gHalo.Y_NORTH[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
        gHalo.Y_NORTH[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
        gHalo.Y_NORTH[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
        // printf("BX=%u BY=%u | NORTH[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[2], pop[5], pop[6]);
    }

    if (tz == 0)
    {
        // BACK FACE
        gHalo.Z_BACK[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[6];
        gHalo.Z_BACK[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[10];
        gHalo.Z_BACK[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[12];
        gHalo.Z_BACK[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[15];
        gHalo.Z_BACK[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[17];
        gHalo.Z_BACK[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
        gHalo.Z_BACK[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
        gHalo.Z_BACK[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
        gHalo.Z_BACK[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
        // printf("BX=%u BY=%u | NORTH[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[2], pop[5], pop[6]);
    }

    if (tz == (blockDim.z - 1))
    {
        // FRONT FACE
        gHalo.Z_FRONT[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[5];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[9];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[11];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[16];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[18];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
        gHalo.Z_FRONT[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
        // printf("BX=%u BY=%u | NORTH[%u,%u]: %.5f %.5f %.5f\n", bx, by, tx, ty, pop[2], pop[5], pop[6]);
    }
}

#endif