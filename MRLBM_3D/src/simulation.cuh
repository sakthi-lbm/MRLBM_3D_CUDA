#pragma once

#include "solve_mrlbm.cuh"

void setup_environment();
void initialize_simulation(Simulation &sim);
void compute_active_blocks(Simulation &sim);
void find_active_blocks_3d(const nodeVar &fMom, std::vector<int> &h_active_blocks);
void finalize_simulation(Simulation &sim);
void restart_simulation(Simulation &sim);
void streaming(Simulation &sim, int iter);
void collision(Simulation &sim, int iter);
void boundary_treatment(Simulation &sim, int iter);
void post_streaming_pipeline(Simulation &sim, int iter);
void post_collision_pipeline(Simulation &sim, int iter);
void post_step_pipeline(Simulation &sim, int iter);
void save_checkpoint(Simulation &sim, int iter);