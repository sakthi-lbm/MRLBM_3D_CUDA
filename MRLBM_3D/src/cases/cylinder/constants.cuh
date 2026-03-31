#pragma once

#include "../../config.h"
#include LATTICE_PROPERTIES

#define CYLINDER

#define X_PERIODIC 0 // or 0
#define Y_PERIODIC 1 // or 0
#define Z_PERIODIC 1 // or 0

#define CONVECTIVE_OUTLET 1
constexpr bool triangular = true;

constexpr MassBC MASS_CONSERV = MassBC::Equilibrium;
constexpr MassBC BCF_MASS_CONSERV = MassBC::Equilibrium;

constexpr int BLOCK_SIZE = 4; // Maxmum block based on the register load

constexpr int D = 16;    // Diameter of the cylinder
constexpr int R = D / 2; // radius of the cylinder
constexpr real D_WALL = toReal(D);
constexpr real R_WALL = 0.5 * D_WALL;

constexpr int LW = 4 * D;  // inlet from cylinder
constexpr int LE = 16 * D; // outlet from cylinder
constexpr int LN = 4 * D;  // top wall from cylinder (y-dir)
constexpr int LS = LN;     // bottom wall from cylinder

constexpr int NX = LW + D + LE; // size x of the grid
constexpr int NY = LN + D + LS; // size y of the grid
constexpr int NZ = 5 * D;       // size z of the grid in one GPU

constexpr int N_OUTLET = NY * NZ;

constexpr real XC = LW + 0.5 * (D - 1); // Center of the cylinder Xc
constexpr real YC = LS + 0.5 * (D - 1); // Center of the cylinder yc

constexpr real RE = 100;      // Reynolds number
constexpr real U_MAX = 0.1;   // lattice characteristic velocity
constexpr real RHO_0 = 1.0;   // Free-stream density
constexpr real delta_t = 1.0; // lattice time-step

constexpr real delx = 1.0; // sqrt (dx2 + dy2) for interpolation

constexpr real UXP_INNER = 0.0;
constexpr real UYP_INNER = 0.0;
constexpr real UZP_INNER = 0.0;

constexpr real VISC = U_MAX * D_WALL / RE;
constexpr real TAU = 0.5 + 3.0 * VISC;
constexpr real OMEGA = 1.0 / TAU;

constexpr real VEL_NORM = U_MAX;

inline real rho_infty = 0.0f;

// ---------- RUNTIME CONSTANTS (HOST) ----------
inline real h_Hxx[Q] = {0};
inline real h_Hyy[Q] = {0};
inline real h_Hzz[Q] = {0};
inline real h_Hxy[Q] = {0};
inline real h_Hxz[Q] = {0};
inline real h_Hyz[Q] = {0};

inline real h_sumUx = 0.0f;
inline real h_UCONV = 1.0 * U_MAX;

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
