#ifndef RESTART_IO_CUH
#define RESTART_IO_CUH

#include "solver/mlbm.cuh"

void write_checkpoint(nodeVar &hMom, haloData &fHalo, int iter);
int read_checkpoint(nodeVar &hMom, haloData &fHalo);

void write_moments(std::ofstream &file, nodeVar &hMom);
void write_halo(std::ofstream &file, haloData &fHalo);

void read_moments(std::ifstream &file, nodeVar &hMom);
void read_halo(std::ifstream &file, haloData &fHalo);



#endif // RESTART_IO_CUH