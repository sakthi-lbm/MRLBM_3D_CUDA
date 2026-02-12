#ifndef SAVE_DATA_H
#define SAVE_DATA_H

#include "all_headers.h"
#include <algorithm>
#include<vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream> // std::cout, std::fixed
#include <iomanip>  // std::setprecision
#include "globalStructs.h"

void write_master_pvd();
void write_vtk_binary(nodeVar h_fMom, int timestep);
void write_vti(nodeVar data, int timestep);
void write_vti_cylinder(nodeVar data, int timestep);

#endif