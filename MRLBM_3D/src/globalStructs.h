#ifndef GLOBAL_STRUCTS_H
#define GLOBAL_STRUCTS_H

#include "config.h"

struct nodeVar
{
    nodeType_t *nodeType;
    real *rho;
    real *ux;
    real *uy;
    real *mxx;
    real *myy;
    real *mxy;
};

struct cylinderVar
{
    size_t *boundaryList;   // size NB
    binary_t *incomings; // size NB x Q
    binary_t *outgoings; // size NB x Q
};

struct haloData
{
    real *X_WEST;
    real *X_EAST;
    real *Y_SOUTH;
    real *Y_NORTH;
};

#endif // GLOBAL_STRUCTS_H