#ifndef STAT_HEADER_H
#define STAT_HEADER_H

#include "../solver/initializeLBM_inline.cuh"

void launch_incoming_force_kernal(const nodeVar &dMom, const cylinderVar &cylinder, const int iter);
void launch_outgoing_force_kernal(const nodeVar &dMom, const cylinderVar &cylinder, const int iter);

__global__ void compute_incoming_force_mass_kernal(const nodeVar dMom, cylinderVar cylinder);
__global__ void compute_outgoing_force_mass_kernal(const nodeVar dMom, cylinderVar cylinder);

#endif // STAT_HEADER_H