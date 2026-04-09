#ifndef STAT_HEADER_H
#define STAT_HEADER_H

#include "../config.h"

#include CASE_BOUNDARY

__device__ __forceinline__ void compute_forces_mass(const uint32_t dir_mask, const real *pop,
                                                    real &Fx, real &Fy, real &Fz, real &m)
{
    Fx = toReal(0.0);
    Fy = toReal(0.0);
    Fz = toReal(0.0);
    m = toReal(0.0);

    uint32_t mask = dir_mask;
    while (mask)
    {
        const int q = __ffs(mask) - 1;
        mask &= mask - 1;

        const real f = pop[q];
        Fx += f * toReal(d_cx[q]);
        Fy += f * toReal(d_cy[q]);
        Fz += f * toReal(d_cz[q]);
        m += f;
    }
}

__device__ __forceinline__ void compute_node_force(const nodeVar &dMom,
                                                   const size_t idx,
                                                   const uint32_t mask,
                                                   real &Fx,
                                                   real &Fy,
                                                   real &Fz,
                                                   real &m)
{
    const real rho = RHO_0 + dMom.rho[idx];
    const real ux = dMom.ux[idx];
    const real uy = dMom.uy[idx];
    const real uz = dMom.uz[idx];

    const real mxx = dMom.mxx[idx];
    const real myy = dMom.myy[idx];
    const real mzz = dMom.mzz[idx];
    const real mxy = dMom.mxy[idx];
    const real mxz = dMom.mxz[idx];
    const real myz = dMom.myz[idx];

    real pop[Q];

    pop_reconstruction(rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz, pop);

    compute_forces_mass(mask, pop, Fx, Fy, Fz, m);
}

__device__ __forceinline__ void compute_force_generic(const nodeVar &dMom,
                                                      const unsigned int tid,
                                                      const size_t *boundaryList,
                                                      const size_t *bcfluidList,
                                                      const uint32_t *boundaryMask,
                                                      const int NB,
                                                      const int NB_FLUID,
                                                      const MaskType masktype,
                                                      real &Fx, real &Fy, real &Fz, real &m)
{
    if (tid < NB)
    {
        size_t idx = boundaryList[tid];
        uint32_t mask = boundaryMask[tid];

        compute_node_force(dMom, idx, mask, Fx, Fy, Fz, m);
    }
    else if (tid < NB + NB_FLUID)
    {
        int fid = tid - NB;
        size_t idx = bcfluidList[fid];

        nodeType_t tag = getIndex(dMom.nodeType[idx]);

        uint32_t mask;
        if (masktype == INCOMING)
        {
            mask = d_incomingMask_bcfluid[tag];
        }
        else
        {
            mask = d_outgoingMask_bcfluid[tag];
        }

        compute_node_force(dMom, idx, mask, Fx, Fy, Fz, m);
    }
}

__device__ __forceinline__ real bilinear_interpolation(const real x, const real y, const real z,
                                                       const real *__restrict__ variable_array)
{

    const int x0 = floor(x);
    const int y0 = floor(y);

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const int zg = toInt(z);

    // The dimensions and indices remain the same
    const real xd = x - toReal(x0);
    const real yd = y - toReal(y0);

    // Calculate block and thread indices for the four corners
    const size_t tx0 = x0 % BLOCK_THREAD_X;
    const size_t tx1 = x1 % BLOCK_THREAD_X;
    const size_t ty0 = y0 % BLOCK_THREAD_Y;
    const size_t ty1 = y1 % BLOCK_THREAD_Y;
    const size_t tz = zg % BLOCK_THREAD_Z;

    const size_t bx0 = x0 / BLOCK_THREAD_X;
    const size_t bx1 = x1 / BLOCK_THREAD_X;
    const size_t by0 = y0 / BLOCK_THREAD_Y;
    const size_t by1 = y1 / BLOCK_THREAD_Y;
    const size_t bz = zg / BLOCK_THREAD_Z;

    // Retrieve the values from the passed array at the four corners
    const real q00 = variable_array[IDX_BLOCK(tx0, ty0, tz, bx0, by0, bz)];
    const real q10 = variable_array[IDX_BLOCK(tx1, ty0, tz, bx1, by0, bz)];
    const real q01 = variable_array[IDX_BLOCK(tx0, ty1, tz, bx0, by1, bz)];
    const real q11 = variable_array[IDX_BLOCK(tx1, ty1, tz, bx1, by1, bz)];

    // Perform the bilinear interpolation
    const real q0 = (toReal(1.0) - xd) * q00 + xd * q10;
    const real q1 = (toReal(1.0) - xd) * q01 + xd * q11;

    const real interp_val = (toReal(1.0) - yd) * q0 + yd * q1;

    return interp_val;
}

__device__ __forceinline__ real surface_pressure_extrapolation(const real xw, const real yw,
                                                               const real x1, const real y1,
                                                               const real x2, const real y2,
                                                               const real x3, const real y3,
                                                               const real rho1, const real rho2, const real rho3)
{
    // pressure interpolation
    const real xc = toReal(XC);
    const real yc = toReal(YC);

    const real xw_diff = xw - xc;
    const real yw_diff = yw - yc;

    const real x1_diff = x1 - xc;
    const real y1_diff = y1 - yc;

    const real x2_diff = x2 - xc;
    const real y2_diff = y2 - yc;

    const real x3_diff = x3 - xc;
    const real y3_diff = y3 - yc;

    const real rw2 = xw_diff * xw_diff + yw_diff * yw_diff;
    const real r12 = x1_diff * x1_diff + y1_diff * y1_diff;
    const real r22 = x2_diff * x2_diff + y2_diff * y2_diff;
    const real r32 = x3_diff * x3_diff + y3_diff * y3_diff;

    const real rw = sqrt(rw2);
    const real r1 = sqrt(r12);
    const real r2 = sqrt(r22);
    const real r3 = sqrt(r32);

    const real denom = (r1 - r2) * (r1 - r3) * (r2 - r3);

    const real p1 = rho1 * cs2;
    const real p2 = rho2 * cs2;
    const real p3 = rho3 * cs2;

    const real a0 = (r1 * r3 * p2 * (r3 - r1) + (r2 * r2) * (r3 * p1 - r1 * p3) + r2 * ((r1 * r1) * p3 - (r3 * r3) * p1)) / denom;
    const real a1 = ((r3 * r3) * (p1 - p2) + (r1 * r1) * (p2 - p3) + (r2 * r2) * (p3 - p1)) / denom;
    const real a2 = (r3 * (p2 - p1) + r2 * (p1 - p3) + r1 * (p3 - p2)) / denom;

    const real pressure = a0 + a1 * rw + a2 * (rw * rw);

    return pressure;
}

__device__ __forceinline__ void compute_surface_pressure_node(const nodeVar &dMom,
                                                     const size_t idx,
                                                     const real unit_nx,
                                                     const real unit_ny,
                                                     real &ps_out)
{
    unsigned int x, y, z;
    GlobalIndexToXYZ(idx, x, y, z);

    // wall point
    const real rmax = toReal(0.5) * D_WALL;
    const real xw = XC + rmax * unit_nx;
    const real yw = YC + rmax * unit_ny;

    // stencil points
    const real x1 = xw + delx * unit_nx;
    const real y1 = yw + delx * unit_ny;
    const real rho1 = RHO_0 + bilinear_interpolation(x1, y1, z, dMom.rho);

    const real x2 = xw + toReal(2.0) * delx * unit_nx;
    const real y2 = yw + toReal(2.0) * delx * unit_ny;
    const real rho2 = RHO_0 + bilinear_interpolation(x2, y2, z, dMom.rho);

    const real x3 = xw + toReal(3.0) * delx * unit_nx;
    const real y3 = yw + toReal(3.0) * delx * unit_ny;
    const real rho3 = RHO_0 + bilinear_interpolation(x3, y3, z, dMom.rho);

    ps_out = surface_pressure_extrapolation(xw, yw, x1, y1, x2, y2, x3, y3, rho1, rho2, rho3);
}

inline void write_forces(const int iter)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "forces.dat");
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc); // delete contents
        first_call = false;
    }

    std::ofstream forcefile(filename, std::ios::app);

    if (forcefile.is_open())
    {
        forcefile << std::setprecision(16) << std::fixed
                  << iter << " " << h_TotalFx << " " << h_TotalFy << " " << h_TotalFz << std::endl;
    }
}

inline void write_mass_flux(const int iter)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "mass_flux.dat");
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc); // delete contents
        first_call = false;
    }

    std::ofstream massfile(filename, std::ios::app);

    if (massfile.is_open())
    {
        massfile << std::setprecision(16) << std::fixed << iter << " " << h_Totalm << std::endl;
    }
}

inline void write_average_inlet_density(const int iter)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "inlet_density.dat");
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc); // delete contents
        first_call = false;
    }

    std::ofstream densityfile(filename, std::ios::app);

    if (densityfile.is_open())
    {
        densityfile << std::setprecision(16) << std::fixed
                    << iter << " "
                    << h_rho_inlet << " "
                    << h_rho_inlet_average << std::endl;
    }
}

#endif // STAT_HEADER_H