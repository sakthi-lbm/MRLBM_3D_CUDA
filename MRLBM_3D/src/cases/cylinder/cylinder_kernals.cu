#include "cylinder_kernals.cuh"

__global__ void apply_bc_cylinder(const int NB, const nodeType_t NODE_TYPE, const cylinderVar &cylinder,
                                  nodeVar dMom, const real UX_PRIME, const real UY_PRIME, const real UZ_PRIME,
                                  const real D_WALL, const int iter)
{

    // boundary index
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= NB)
        return;

    // global index loaded from the boundary list
    const size_t idx = cylinder.boundaryList[i];

    // converting global index to the gloabl coordinates
    unsigned int x, y, z;
    GlobalIndexToXYZ(idx, x, y, z);

    const nodeType_t nodeType = dMom.nodeType[idx];
    real rho = RHO_0 + dMom.rho[idx]; // Incoming density rhoI
    real ux = dMom.ux[idx];
    real uy = dMom.uy[idx];
    real uz = dMom.uz[idx];
    real mxx = dMom.mxx[idx]; // Incoming Moment mxxI
    real myy = dMom.myy[idx]; // Incoming Moment myyI
    real mzz = dMom.mzz[idx]; // Incoming Moment mzzI
    real mxy = dMom.mxy[idx]; // Incoming Moment mxyI
    real mxz = dMom.mxz[idx]; // Incoming Moment mxzI
    real myz = dMom.myz[idx]; // Incoming Moment myzI

    if (nodeType >= NODE_TYPE && nodeType < (NODE_TYPE + NB))
    {
        const real delta = cylinder.delta_w[i];
        const real unit_nx = cylinder.unit_nx[i];
        const real unit_ny = cylinder.unit_ny[i];
        const real xw = XC + toReal(0.5) * D_WALL * unit_nx;
        const real yw = YC + toReal(0.5) * D_WALL * unit_ny;
        const real zw = toReal(z);

        curved_boundary_condition_rotated(unit_nx, unit_ny, delta, x, y, z, xw, yw, zw, cylinder, nodeType, dMom,
                                          rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz,
                                          UX_PRIME, UY_PRIME, UZ_PRIME, NODE_TYPE, iter);
    }

    // writing  moments into global memory (being done only for cylinder block)
    dMom.rho[idx] = rho - RHO_0;
    dMom.ux[idx] = ux;
    dMom.uy[idx] = uy;
    dMom.uz[idx] = uz;
    dMom.mxx[idx] = mxx;
    dMom.myy[idx] = myy;
    dMom.mzz[idx] = mzz;
    dMom.mxy[idx] = mxy;
    dMom.mxz[idx] = mxz;
    dMom.myz[idx] = myz;
}

__device__ void cylinder_boundary_moments(nodeType_t nodeType, cylinderVar &cylinder, nodeVar &dMom, real *pop,
                                          real &rho, real &ux, real &uy, real &uz,
                                          real &mxx, real &myy, real &mzz,
                                          real &mxy, real &mxz, real &myz)
{
    const int NB = cylinder.NB;
    if (nodeType >= INNER_NODE && nodeType < (INNER_NODE + NB))
    {
        const nodeType_t id = nodeType - INNER_NODE;
        const real unit_nx = cylinder.unit_nx[id];
        const real unit_ny = cylinder.unit_ny[id];
        const uint32_t incomingMask = cylinder.incomingMask[id];

        evaluate_incoming_moments_rotated(unit_nx, unit_ny, incomingMask, pop, rho, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (triangular && nodeType >= (BCFLUID_NODE + 0) && nodeType < (BCFLUID_NODE + 256))
    {
        const nodeType_t nodeTag = nodeType - BCFLUID_NODE;
        const uint32_t incomingMask = d_incomingMask_bcfluid[nodeTag];

        fluid_boundary_condition(nodeTag, incomingMask, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else if (!Z_PERIODIC && nodeType >= (BCSOLID_NODE + 0) && nodeType < (BCSOLID_NODE + 256))
    {
        const nodeType_t nodeTag = nodeType - BCSOLID_NODE;

        bcsolid_boundary_condition(nodeTag, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        boundary_condition(nodeType, dMom, pop, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
}