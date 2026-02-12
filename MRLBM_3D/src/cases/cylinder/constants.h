#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../../config.h"
#include "../../solver/latticeProperties.cuh"

#define CYLINDER

#define X_PERIODIC 0 // or 0
#define Y_PERIODIC 1 // or 0

constexpr bool rotated_coordinates = true;
constexpr bool triangular = true;

constexpr MassBC MASS_CONSERV = MassBC::Equilibrium;
constexpr MassBC BCF_MASS_CONSERV = MassBC::Equilibrium;

constexpr int BLOCK_SIZE = 16; // Maxmum block based on the register load

constexpr int D = 32;                 // Diameter of the cylinder
constexpr int R = D / 2;              // radius of the cylinder
constexpr int L_UP = 15 * D;          // Upstream length from the cylinder
constexpr int L_DOWN = 30 * D;        // Downstream length from the cylinder
constexpr int L_TOP = 10 * D;         // Length of Top wall from the cylinder
constexpr int L_BOT = L_TOP;          // Length of Bottom wall from the cylinder
constexpr int NX = L_UP + D + L_DOWN; // Length of domain
constexpr int NY = L_BOT + D + L_TOP; // Height of the domain

constexpr real D_WALL = toReal(D);
constexpr real R_WALL = 0.5 * D_WALL;

constexpr real XC = L_UP + 0.5 * (D - 1);  // Center of the cylinder Xc
constexpr real YC = L_BOT + 0.5 * (D - 1); // Center of the cylinder yc

constexpr real RE = 100;      // Reynolds number
constexpr real U_MAX = 0.1;  // lattice characteristic velocity
constexpr real RHO_0 = 1.0;   // Free-stream density
constexpr real delta_t = 1.0; // lattice time-step

constexpr real delx = 1.0; // sqrt (dx2 + dy2) for interpolation

constexpr real UXP_CYLINDER = 0.0;
constexpr real UYP_CYLINDER = 0.0;

constexpr real VISC = U_MAX * D_WALL / RE;
constexpr real TAU = 0.5 + 3.0 * VISC;
constexpr real OMEGA = 1.0 / TAU;

inline int NB = 0;   // Number of cylinder boundary points on cylinder
inline int NBCF = 0; // Number of bc fluid points due to triangular grid

inline real rho_infty = 0.0f;

inline binary_t h_incomings_bcfluid[4][Q] = {0};
inline binary_t h_outgoings_bcfluid[4][Q] = {0};

inline real h_TotalFx = 0.0f;
inline real h_TotalFy = 0.0f;
inline real h_Totalm = 0.0f;

inline real h_min_avg = 0.0f;
inline real h_mout_avg = 0.0f;

// ---------- RUNTIME CONSTANTS (DEVICE) ----------
extern __device__ __constant__ real d_Hxx[Q];
extern __device__ __constant__ real d_Hyy[Q];
extern __device__ __constant__ real d_Hxy[Q];
extern __device__ __constant__ int d_NB;
extern __device__ __constant__ int d_NBCF;

extern __device__ __constant__ binary_t d_incomings_bcfluid[4][Q];
extern __device__ __constant__ binary_t d_outgoings_bcfluid[4][Q];

extern __device__ real d_TotalFx;
extern __device__ real d_TotalFy;
extern __device__ real d_Totalm;

#endif // CONSTANTS_H