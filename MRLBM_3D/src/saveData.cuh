#ifndef SAVE_DATA_H
#define SAVE_DATA_H

#include <vector>
#include "restart_io.cuh"

void write_master_pvd();
void write_vti_3d(nodeVar data, int timestep);

#endif // SAVE_DATA_H