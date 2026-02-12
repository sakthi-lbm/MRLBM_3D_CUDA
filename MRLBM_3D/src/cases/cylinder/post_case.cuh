#ifndef POST_CASE_H
#define POST_CASE_H
#include <vector>
#include <iomanip>
#include "../../all_headers.h"

#ifdef CYLINDER

inline void calculate_write_inlet_average_density(const nodeVar &fMom, const int iter)
{
    //===================== calculating average inlet density ==================================
    real rho_inlet = 0.0;
    for (int y = 0; y < NY; y++)
    {
        const int x = 0;
        const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                     y % BLOCK_THREAD_Y,
                                     x / BLOCK_THREAD_X,
                                     y / BLOCK_THREAD_Y);
        const real rho = RHO_0 + fMom.rho[idx];
        rho_inlet += rho;
    }
    rho_inlet /= NY;
    rho_infty = rho_inlet; // Storing inlet average density  as rho_infty to a global variable

    std::string filename = construct_path(PATH_FILES, ID_SIM, "rho_inlet.dat");
    // clearing the file for the first time and  then append
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc);
        first_call = false;
        clear << "iter \t" << "rho_inlet" << std::endl;
    }

    std::ofstream rho_file(filename, std::ios::app);
    if (!rho_file.is_open())
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }
    rho_file << std::setprecision(8) << std::fixed << iter << " " << rho_inlet << std::endl;
    rho_file.close();
    //===================================================================================================

    //===================== calculating time averaged inlet and outlet mass flowrate ==================================
    real m_in = 0.0;
    real m_out = 0.0;
    // INLET
    for (int y = 0; y < NY; y++)
    {
        const int x = 0;
        const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                     y % BLOCK_THREAD_Y,
                                     x / BLOCK_THREAD_X,
                                     y / BLOCK_THREAD_Y);
        const real rho = RHO_0 + fMom.rho[idx];
        const real ux = fMom.ux[idx];
        m_in += rho * ux;
    }

    // OUTLET
    for (int y = 0; y < NY; y++)
    {
        const int x = NX - 1;
        const size_t idx = IDX_BLOCK(x % BLOCK_THREAD_X,
                                     y % BLOCK_THREAD_Y,
                                     x / BLOCK_THREAD_X,
                                     y / BLOCK_THREAD_Y);
        const real rho = RHO_0 + fMom.rho[idx];
        const real ux = fMom.ux[idx];
        m_out += rho * ux;
    }

    // Averaging
    const int counter = iter - STAT_START;
    h_min_avg = (h_min_avg * counter + m_in) / (counter + 1);
    h_mout_avg = (h_mout_avg * counter + m_out) / (counter + 1);
    const real Error_m = (h_min_avg - h_mout_avg) / h_min_avg;

    std::string filename2 = construct_path(PATH_FILES, ID_SIM, "mass_conservation.dat");
    static bool first_call2 = true;
    if (first_call2)
    {
        std::ofstream clear(filename2, std::ios::trunc);
        first_call2 = false;
        clear << "iter \t" << "m_in \t" << "m_out \t" << "m_in_avg \t" << "m_out_avg \t" << "Error" << std::endl;
    }

    std::ofstream mass_conserv_file(filename2, std::ios::app);
    if (!mass_conserv_file.is_open())
    {
        std::cerr << "Error: cannot open " << filename2 << std::endl;
        return;
    }
    mass_conserv_file << std::setprecision(8) << std::fixed << iter << " " << m_in << " " << m_out << " "
                      << h_min_avg << " " << h_mout_avg << " " << Error_m << std::endl;

    rho_file.close();
    //===================================================================================================
}

inline void calculate_write_theta(const nodeVar &fMom, const cylinderVar &cylinder)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "theta.dat");
    std::ofstream theta_file(filename);
    if (!theta_file.is_open())
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }
    for (int i = 0; i < NB; i++)
    {
        const size_t idx = cylinder.boundaryList[i];
        unsigned int x, y;
        GlobalIndexToXY(idx, x, y);

        const real x_diff = toReal(x - XC);
        const real y_diff = toReal(y - YC);

        real theta = atan2f(y_diff, x_diff);
        if (theta < 0)
        {
            theta += 2.0 * PI;
        }

        theta_file << std::setprecision(12) << " " << theta;
    }
    theta_file << std::endl;
    theta_file.close();
}

inline real pressure_bilinear_interpolation(real x, real y, size_t x0, size_t y0, size_t x1, size_t y1,
                                            real *variable_array)
{
    // The dimensions and indices remain the same
    const real xd = x - toReal(x0);
    const real yd = y - toReal(y0);

    // Calculate block and thread indices for the four corners
    const size_t tx0 = x0 % BLOCK_THREAD_X;
    const size_t tx1 = x1 % BLOCK_THREAD_X;
    const size_t ty0 = y0 % BLOCK_THREAD_Y;
    const size_t ty1 = y1 % BLOCK_THREAD_Y;

    const size_t bx0 = x0 / BLOCK_THREAD_X;
    const size_t bx1 = x1 / BLOCK_THREAD_X;
    const size_t by0 = y0 / BLOCK_THREAD_Y;
    const size_t by1 = y1 / BLOCK_THREAD_Y;

    // Retrieve the values from the passed array at the four corners
    const real q00 = RHO_0 + variable_array[IDX_BLOCK(tx0, ty0, bx0, by0)];
    const real q10 = RHO_0 + variable_array[IDX_BLOCK(tx1, ty0, bx1, by0)];
    const real q01 = RHO_0 + variable_array[IDX_BLOCK(tx0, ty1, bx0, by1)];
    const real q11 = RHO_0 + variable_array[IDX_BLOCK(tx1, ty1, bx1, by1)];

    // Perform the bilinear interpolation
    const real q0 = (toReal(1.0) - xd) * q00 + xd * q10;
    const real q1 = (toReal(1.0) - xd) * q01 + xd * q11;

    const real interp_val = (toReal(1.0) - yd) * q0 + yd * q1;

    return interp_val;
}
inline real surface_pressure_extrapolation(const real xw, const real yw,
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

inline void calculate_write_pressure(const nodeVar &fMom, const cylinderVar &cylinder, const int iter)
{
    std::string filename = construct_path(PATH_FILES, ID_SIM, "pressure.dat");
    static bool first_call = true;
    if (first_call)
    {
        std::ofstream clear(filename, std::ios::trunc);
        first_call = false;
    }

    std::ofstream pressure_file(filename, std::ios::app);
    if (!pressure_file.is_open())
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }

    for (int i = 0; i < NB; i++)
    {
        const size_t idx = cylinder.boundaryList[i];
        unsigned int x, y;
        GlobalIndexToXY(idx, x, y);

        // cylinder center
        const real xc = toReal(XC);
        const real yc = toReal(YC);

        // Boundary node location
        const real xb = toReal(x);
        const real yb = toReal(y);

        // unit normal calculation
        const real x_diff = xb - xc;
        const real y_diff = yb - yc;
        const real radius = sqrt(x_diff * x_diff + y_diff * y_diff);
        const real inv_radius = toReal(1.0) / radius;

        const real unit_nx = x_diff * inv_radius;
        const real unit_ny = y_diff * inv_radius;

        // wall point location (cylinder)
        const real rmax = toReal(0.5) * D;
        const real xw = xc + rmax * unit_nx;
        const real yw = yc + rmax * unit_ny;

        // distance between the wall point and boundary node
        const real delta_x = xw - xb;
        const real delta_y = yw - yb;
        const real delta = sqrt(delta_x * delta_x + delta_y * delta_y);

        // First reference fluid point and pressure calculation
        real xf = xw + delx * unit_nx;
        real yf = yw + delx * unit_ny;

        const real x1 = xf;
        const real y1 = yf;

        size_t int_xf = toSize_t(xf);
        size_t int_yf = toSize_t(yf);

        size_t int_xfp1 = int_xf + 1;
        size_t int_yfp1 = int_yf + 1;

        const real rho1 = pressure_bilinear_interpolation(xf, yf, int_xf, int_yf, int_xfp1, int_yfp1, fMom.rho);

        // Second reference fluid point and pressure calculation
        xf = xw + toReal(2.0) * delx * unit_nx;
        yf = yw + toReal(2.0) * delx * unit_ny;

        const real x2 = xf;
        const real y2 = yf;

        int_xf = toSize_t(xf);
        int_yf = toSize_t(yf);

        int_xfp1 = int_xf + 1;
        int_yfp1 = int_yf + 1;

        const real rho2 = pressure_bilinear_interpolation(xf, yf, int_xf, int_yf, int_xfp1, int_yfp1, fMom.rho);

        // Third reference fluid point and pressure calculation
        xf = xw + toReal(3.0) * delx * unit_nx;
        yf = yw + toReal(3.0) * delx * unit_ny;

        const real x3 = xf;
        const real y3 = yf;

        int_xf = toSize_t(xf);
        int_yf = toSize_t(yf);

        int_xfp1 = int_xf + 1;
        int_yfp1 = int_yf + 1;

        const real rho3 = pressure_bilinear_interpolation(xf, yf, int_xf, int_yf, int_xfp1, int_yfp1, fMom.rho);

        // surface pressure extrapolation
        const real ps = surface_pressure_extrapolation(xw, yw, x1, y1, x2, y2, x3, y3,
                                                       rho1, rho2, rho3);
        pressure_file << std::setprecision(12) << " " << ps;
    }
    pressure_file << std::endl;
    pressure_file.close();
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
        forcefile << std::setprecision(16) << std::fixed << iter << " " << h_TotalFx << " " << h_TotalFy << std::endl;
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

__device__ inline void compute_incoming_forces_mass(const nodeType_t nodeTag, const cylinderVar &cylinder, const real *pop,
                                                    real &Fx_in, real &Fy_in, real &m_in)
{
    Fx_in = toReal(0.0);
    Fy_in = toReal(0.0);
    m_in = toReal(0.0);
    for (size_t q = 0; q < Q; q++)
    {
        if (cylinder.incomings[idxBoundPop(nodeTag, q)] == 1)
        {
            const real cx = toReal(d_cx[q]);
            const real cy = toReal(d_cy[q]);
            Fx_in += pop[q] * cx;
            Fy_in += pop[q] * cy;
            m_in += pop[q];
        }
    }
}

__device__ inline void compute_incoming_forces_mass_bcfluid(const nodeType_t nodeTag, const real *pop,
                                                            real &Fx_in, real &Fy_in, real &m_in)
{
    Fx_in = toReal(0.0);
    Fy_in = toReal(0.0);
    m_in = toReal(0.0);
    for (size_t q = 0; q < Q; q++)
    {
        if (d_incomings_bcfluid[nodeTag][q] == 1)
        {
            const real cx = toReal(d_cx[q]);
            const real cy = toReal(d_cy[q]);
            Fx_in += pop[q] * cx;
            Fy_in += pop[q] * cy;
            m_in += pop[q];
        }
    }
}

__device__ inline void compute_outgoing_forces_mass(const nodeType_t nodeTag, const cylinderVar &cylinder, const real *pop,
                                                    real &Fx_out, real &Fy_out, real &m_out)
{
    Fx_out = toReal(0.0);
    Fy_out = toReal(0.0);
    m_out = toReal(0.0);
    for (size_t q = 0; q < Q; q++)
    {
        if (cylinder.outgoings[idxBoundPop(nodeTag, q)] == 1)
        {
            const real cx = toReal(d_cx[q]);
            const real cy = toReal(d_cy[q]);
            Fx_out += pop[q] * cx;
            Fy_out += pop[q] * cy;
            m_out += pop[q];
        }
    }
}

__device__ inline void compute_outgoing_forces_mass_bcfluid(const nodeType_t nodeTag, const real *pop,
                                                            real &Fx_out, real &Fy_out, real &m_out)
{
    Fx_out = toReal(0.0);
    Fy_out = toReal(0.0);
    m_out = toReal(0.0);
    for (size_t q = 0; q < Q; q++)
    {
        if (d_outgoings_bcfluid[nodeTag][q] == 1)
        {
            const real cx = toReal(d_cx[q]);
            const real cy = toReal(d_cy[q]);
            Fx_out += pop[q] * cx;
            Fy_out += pop[q] * cy;
            m_out += pop[q];
        }
    }
}


#endif

#endif