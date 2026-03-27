#include <iostream>
#include <fstream>
#include <numeric>

#include "main.cuh"

int main()
{
    //==================================== MRLBM =====================================
    setup_environment();

    Simulation sim;
    setup_case(sim);
    
    initialize_simulation(sim);
    compute_active_blocks(sim);
    restart_simulation(sim);

    for (int iter = sim.start_iter; iter <= MAX_ITER; iter++)
    {
        streaming(sim, iter);

        boundary_treatment(sim, iter);

        // post_streaming_pipeline(sim, iter);

        collision(sim, iter);

        // post_collision_pipeline(sim, iter);

        post_step_pipeline(sim, iter);
    }

    finalize_simulation(sim);

    return 0;
}