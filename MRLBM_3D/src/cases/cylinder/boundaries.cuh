#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include "../../nodeTypeMap.h"
#include "constants.h"
#include "../../globalStructs.h"

__host__ __device__ inline nodeType_t boundary_definitions(const int x, const int y, const int z)
{
    // Determine boundary flags based on position
    bool isW = (x == 0);
    bool isE = (x == NX - 1);
    bool isS = (y == 0);
    bool isN = (y == NY - 1);
    bool isB = (z == 0);
    bool isF = (z == NZ - 1);

    // Apply Periodicity: if a direction is periodic, it's treated as BULK (no boundary)
    #if X_PERIODIC
        isW = false; isE = false;
    #endif
    #if Y_PERIODIC
        isS = false; isN = false;
    #endif
    #if Z_PERIODIC
        isB = false; isF = false;
    #endif

    // --- CORNERS (3-way intersection) ---
    if (isN && isW && isF) return NORTH_WEST_FRONT;
    if (isN && isW && isB) return NORTH_WEST_BACK;
    if (isN && isE && isF) return NORTH_EAST_FRONT;
    if (isN && isE && isB) return NORTH_EAST_BACK;
    if (isS && isW && isF) return SOUTH_WEST_FRONT;
    if (isS && isW && isB) return SOUTH_WEST_BACK;
    if (isS && isE && isF) return SOUTH_EAST_FRONT;
    if (isS && isE && isB) return SOUTH_EAST_BACK;

    // --- EDGES (2-way intersection) ---
    if (isN && isW) return NORTH_WEST;
    if (isN && isE) return NORTH_EAST;
    if (isN && isF) return NORTH_FRONT;
    if (isN && isB) return NORTH_BACK;
    
    if (isS && isW) return SOUTH_WEST;
    if (isS && isE) return SOUTH_EAST;
    if (isS && isF) return SOUTH_FRONT;
    if (isS && isB) return SOUTH_BACK;
    
    if (isW && isF) return WEST_FRONT;
    if (isW && isB) return WEST_BACK;
    if (isE && isF) return EAST_FRONT;
    if (isE && isB) return EAST_BACK;

    // --- FACES (1-way intersection) ---
    if (isN) return NORTH;
    if (isS) return SOUTH;
    if (isW) return WEST;
    if (isE) return EAST;
    if (isF) return FRONT;
    if (isB) return BACK;

    // Default
    return BULK;
}


#endif // BOUDARIES_H