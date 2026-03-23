#pragma once

#include "solve_mrlbm.cuh"

void setup_environment();
void initialize_simulation(Simulation &sim);
void finalize_simulation(Simulation &sim);
void restart_simulation(Simulation &sim);
void streaming(Simulation &sim, int iter);
void collision(Simulation &sim, int iter);
void boundary_treatment(Simulation &sim, int iter);
void post_streaming_pipeline(Simulation &sim, int iter);
void post_collision_pipeline(Simulation &sim, int iter);
void post_step_pipeline(Simulation &sim, int iter);
void save_checkpoint(Simulation &sim, int iter);