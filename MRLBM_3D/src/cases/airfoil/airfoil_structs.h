#pragma once

#include "../../boundary/curvedLBM.cuh"

struct airfoilVar
{
    int NB;
    int NB_FLUID;
    int NB_SOLID;

    real *unit_nx; // unit normal to the airfoil
    real *unit_ny; // unit tangential to the airfoil
    real *delta_w; // distance between wall and boundary node

    size_t *boundaryList;   // size NB
    uint32_t *incomingMask; // NB
    uint32_t *outgoingMask; // NB

    size_t *bcfluidList; // size NB_FLUID
    size_t *bcsolidList; // size NB_SOLID

    // airfoil forces
    real d_TotalFx;
    real d_TotalFy;
    real d_TotalFz;
    real d_Totalm;
};

struct airfoilPostProcess
{
    // time averaging
    real *Cp_avg;      // NB × Nz
    real *Cp_rms_avg;  // NB × Nz(RMS)
    real *Cp_span_avg; // NB

    int n_avg = 0;
};