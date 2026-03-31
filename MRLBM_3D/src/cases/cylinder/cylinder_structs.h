#pragma once

#include "../../output/saveData.cuh"

struct boundaryVar
{
    int NB;
    int NB_FLUID;
    int NB_SOLID;

    real *unit_nx; // unit normal to the annulus
    real *unit_ny; // unit tangential to the annulus
    real *delta_w; // distance between wall and boundary node

    size_t *boundaryList;   // size NB
    uint32_t *incomingMask; // NB
    uint32_t *outgoingMask; // NB

    size_t *bcfluidList; // size NB_FLUID
    size_t *bcsolidList; // size NB_SOLID

    // annulus forces
    real d_TotalFx;
    real d_TotalFy;
    real d_TotalFz;
    real d_Totalm;
};

struct cylinderVar
{
    boundaryVar inner;
};

struct cylinderPostProcess
{
    // time averaging
    real *ps_avg;     // NB × Nz
    real *ps_rms_avg; // NB × Nz(RMS)
    int n_avg = 0;
};