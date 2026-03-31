#pragma once

#include "../../config.h"
#include LATTICE_PROPERTIES

#define ANNULUS

#define X_PERIODIC 0 // or 0
#define Y_PERIODIC 0 // or 0
#define Z_PERIODIC 0 // or 0

constexpr bool triangular = true;

constexpr MassBC MASS_CONSERV = MassBC::Equilibrium;
constexpr MassBC BCF_MASS_CONSERV = MassBC::Equilibrium;

constexpr int BLOCK_SIZE = 4; // Maxmum block based on the register load

constexpr int N_GAP_BASE = 4;
constexpr int multiplier = 8;                  // Multiplier to refine the grid
constexpr int N_GAP = N_GAP_BASE * multiplier; // Number of nodes in the gap

constexpr real RADIUS_RATIO = 0.5;   // r_in/r_out
constexpr real ROTATION_RATIO = 0.0; // omega_out / omega_in

// D_out to maintain N_gap for this radius ratio
constexpr int D_out_raw = toInt(2.0 * N_GAP / (1.0 - RADIUS_RATIO));

// D_out to the nearest multiple of BLOCK_SIZE for cuda block layout
constexpr int D_out_aligned = ((D_out_raw + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE; // Diameter of the outer cylinder
constexpr real D_OUT = toReal(D_out_aligned);
constexpr real R_OUT = toReal(0.5) * D_OUT; // radius of the outer cylinder
constexpr real R_IN = RADIUS_RATIO * R_OUT; // radius of the inner cylinder
constexpr real D_IN = toReal(2.0) * R_IN;   // Diameter of the inner cylinder

constexpr int NX_phys = D_out_aligned + 4; // keep minimum 4
constexpr int NY_phys = D_out_aligned + 4;
constexpr int NZ_phys = 256;

constexpr int NX = ((NX_phys + BLOCK_THREAD_X - 1) / BLOCK_THREAD_X) * BLOCK_THREAD_X;
constexpr int NY = ((NY_phys + BLOCK_THREAD_Y - 1) / BLOCK_THREAD_Y) * BLOCK_THREAD_Y;
constexpr int NZ = ((NZ_phys + BLOCK_THREAD_Z - 1) / BLOCK_THREAD_Z) * BLOCK_THREAD_Z;

constexpr int N_OUTLET = NY * NZ;

constexpr real XC = 0.5 * (NX - 1); // Center of the cylinder xc
constexpr real YC = 0.5 * (NY - 1); // Center of the cylinder yc

constexpr real RE = 70;       // Reynolds number
constexpr real U_MAX = 0.03;  // lattice characteristic velocity
constexpr real RHO_0 = 1.0;   // Free-stream density
constexpr real delta_t = 1.0; // lattice time-step

constexpr real delx = 1.0; // sqrt (dx2 + dy2) for interpolation

constexpr real U_i_raw = R_IN; // omega = 1.0
constexpr real U_o_raw = rabs(ROTATION_RATIO) * R_OUT;

constexpr real U_raw_max = (U_i_raw > U_o_raw) ? U_i_raw : U_o_raw;
constexpr real VEL_SCALE = U_MAX / U_raw_max;

constexpr real OMEGA_INNER = VEL_SCALE;
constexpr real OMEGA_OUTER = ROTATION_RATIO * OMEGA_INNER;

constexpr real UXP_INNER = 0.0;
constexpr real UYP_INNER = OMEGA_INNER * R_IN;
constexpr real UZP_INNER = 0.0;

constexpr real UXP_OUTER = 0.0;
constexpr real UYP_OUTER = OMEGA_OUTER * R_OUT;
constexpr real UZP_OUTER = 0.0;

constexpr real U_REL = rabs(UYP_INNER - UYP_OUTER);

constexpr real VEL_NORM = UYP_INNER;              // Velocity norm for non-dimensionalization
constexpr real P_NORM = RHO_0 * RHO_0 * VEL_NORM; // Pressure norm for non-dimensionalization

constexpr real GAP = R_OUT - R_IN;
constexpr real VISC = U_REL * GAP / RE;
constexpr real TAU = 0.5 + as2 * VISC;
constexpr real OMEGA = 1.0 / TAU;

constexpr real ETA = R_IN / R_OUT;
constexpr real Ta = ((1.0 + ETA) * (1.0 + ETA) * (1.0 + ETA) * (1.0 + ETA)) / (4.0 * ETA * ETA) * RE * RE;

inline real rho_infty = 0.0f;

// ---------- RUNTIME CONSTANTS (HOST) ----------
inline real h_Hxx[Q] = {0};
inline real h_Hyy[Q] = {0};
inline real h_Hzz[Q] = {0};
inline real h_Hxy[Q] = {0};
inline real h_Hxz[Q] = {0};
inline real h_Hyz[Q] = {0};

inline real h_sumUx = 0.0f;
inline real h_UCONV = 0.9 * U_MAX;

inline real h_TotalFx = 0.0f;
inline real h_TotalFy = 0.0f;
inline real h_TotalFz = 0.0f;
inline real h_Totalm = 0.0f;

#define MAX_NODE_TAG 256
inline uint32_t h_incomingMask_bcfluid[MAX_NODE_TAG];
inline uint32_t h_outgoingMask_bcfluid[MAX_NODE_TAG];
inline uint32_t h_incomingMask_bcsolid[MAX_NODE_TAG];
inline uint32_t h_outgoingMask_bcsolid[MAX_NODE_TAG];

// ---------- RUNTIME CONSTANTS (DEVICE) ----------
extern __constant__ real d_w[Q];
extern __constant__ int d_cx[Q];
extern __constant__ int d_cy[Q];
extern __constant__ int d_cz[Q];

extern __constant__ real d_Hxx[Q];
extern __constant__ real d_Hyy[Q];
extern __constant__ real d_Hzz[Q];
extern __constant__ real d_Hxy[Q];
extern __constant__ real d_Hxz[Q];
extern __constant__ real d_Hyz[Q];

extern __device__ real d_sumUx;
extern __device__ real d_UCONV;

extern __constant__ uint32_t d_incomingMask_bcfluid[MAX_NODE_TAG];
extern __constant__ uint32_t d_outgoingMask_bcfluid[MAX_NODE_TAG];
extern __constant__ uint32_t d_incomingMask_bcsolid[MAX_NODE_TAG];
extern __constant__ uint32_t d_outgoingMask_bcsolid[MAX_NODE_TAG];

extern __device__ real d_TotalFx;
extern __device__ real d_TotalFy;
extern __device__ real d_TotalFz;
extern __device__ real d_Totalm;
