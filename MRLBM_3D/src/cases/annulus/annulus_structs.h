#pragma once

#include "../../boundary/curvedLBM.cuh"

struct boundaryVar
{
    int NB = 0;
    int NB_FLUID = 0;
    int NB_SOLID = 0;

    real *unit_nx = nullptr;
    real *unit_ny = nullptr;
    real *delta_w = nullptr;

    size_t *boundaryList = nullptr;
    uint32_t *incomingMask = nullptr;
    uint32_t *outgoingMask = nullptr;

    size_t *bcfluidList = nullptr;
    size_t *bcsolidList = nullptr;
};

struct annulusVar
{
    boundaryVar inner;
    boundaryVar outer;
};

struct annulusPostProcess
{
    // time averaging
    real *ps_avg = nullptr;     // NB × Nz
    real *ps_rms_avg = nullptr; // NB × Nz(RMS)
    int n_avg = 0;
};