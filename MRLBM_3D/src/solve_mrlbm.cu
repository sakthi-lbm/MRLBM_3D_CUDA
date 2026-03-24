#include "solve_mrlbm.cuh"

void setup_case(Simulation &sim)
{

// Case  selection
#if CASE_ID == CASE_CYLINDER

    setup_cylinder_case(sim.case_module);

#elif CASE_ID == CASE_AIRFOIL

    setup_airfoil_case(sim.case_module);

#elif CASE_ID == CASE_ANNULUS

    setup_annulus_case(sim.case_module);

#else

#error "Unknown BC_PROBLEM"

#endif
}