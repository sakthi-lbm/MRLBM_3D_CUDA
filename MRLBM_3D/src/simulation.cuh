#pragma once

#include "config.h"
#include CASE_BOUNDARY

struct Simulation; // forward declaration

struct Case
{
    void (*initialize)(Simulation &);
    void (*apply_boundary)(Simulation &, int iter);

    void (*compute_forces)(Simulation &, int iter) = nullptr;
    void (*post_process_step)(Simulation &, int iter) = nullptr;
    void (*finalize)(Simulation &) = nullptr;
};

struct Simulation
{
    nodeVar h_fMom, d_fMom;
    haloData h_fHalo, fHalo_interface, gHalo_interface;

    // case specific
    cylinderVar h_cylinder, d_cylinder;
    Case case_module;

    int start_iter = 0;
};