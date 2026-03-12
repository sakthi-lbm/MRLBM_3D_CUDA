#ifndef GLOBAL_STRUCTS_H
#define GLOBAL_STRUCTS_H

#include "config.h"

struct nodeVar
{
    nodeType_t *nodeType;
    real *rho;
    real *ux;
    real *uy;
    real *uz;
    real *mxx;
    real *mxy;
    real *mxz;
    real *myy;
    real *myz;
    real *mzz;
};

struct cylinderVar
{
    size_t *boundaryList;   // size NB
    uint32_t *incomingMask; // NB
    uint32_t *outgoingMask; // NB

    size_t *bcfluidList;            // size NB_FLUID
    uint32_t *incomingMask_bcfluid; // NB_FLUID
    uint32_t *outgoingMask_bcfluid; // NB_FLUID
};

struct VelocityMoments
{
    real ux;
    real uy;
    real uz;
    real mxx;
    real myy;
    real mzz;
    real mxy;
    real mxz;
    real myz;
};

struct haloData
{
    real *X_WEST;
    real *X_EAST;
    real *Y_SOUTH;
    real *Y_NORTH;
    real *Z_FRONT;
    real *Z_BACK;
};

#endif // GLOBAL_STRUCTS_H