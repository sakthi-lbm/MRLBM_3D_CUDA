#pragma once

#include "nodeTypeMap.cuh"

struct nodeVar
{
    nodeType_t *nodeType;
    real *rho;
    real *ux;
    real *uy;
    real *uz;
    real *mxx;
    real *mxy;
    real *mxz;
    real *myy;
    real *myz;
    real *mzz;
};

struct VelocityMoments
{
    real ux;
    real uy;
    real uz;
    real mxx;
    real myy;
    real mzz;
    real mxy;
    real mxz;
    real myz;
};

struct haloData
{
    real *X_WEST;
    real *X_EAST;
    real *Y_SOUTH;
    real *Y_NORTH;
    real *Z_FRONT;
    real *Z_BACK;
};

struct cylinderVar
{
    int NB;
    int NB_FLUID;
    int NB_SOLID;

    real *unit_nx; // unit normal to the cylinder
    real *unit_ny; // unit tangential to the cylinder
    real *delta_w; // distance between wall and boundary node

    size_t *boundaryList;   // size NB
    uint32_t *incomingMask; // NB
    uint32_t *outgoingMask; // NB

    size_t *bcfluidList; // size NB_FLUID
    size_t *bcsolidList; // size NB_SOLID

    // cylinder forces
    real d_TotalFx;
    real d_TotalFy;
    real d_TotalFz;
    real d_Totalm;
};

struct Simulation; // forward declaration

struct Case
{
    void (*initialize)(Simulation &);
    void (*apply_boundary)(Simulation &, int iter);

    void (*compute_forces)(Simulation &, int iter) = nullptr;
    void (*post_process_step)(Simulation &, int iter) = nullptr;
    void (*finalize)(Simulation &) = nullptr;

    void (*free_case)(Simulation &) = nullptr;
};

struct Profiler
{
    timestep sim_start_time;
    timestep step_start;
    timestep step_end;
    timestep end_time;
    real mlups = 0.0;
};

struct Simulation
{
    nodeVar h_fMom, d_fMom;
    haloData h_fHalo;
    haloData d_fHalo, d_gHalo;

    // case specific
    void *h_caseData = nullptr;
    void *d_caseData = nullptr;

    Case case_module = {};

    int start_iter = 0;

    Profiler profile;
};
