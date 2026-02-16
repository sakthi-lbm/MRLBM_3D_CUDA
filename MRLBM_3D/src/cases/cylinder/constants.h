#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../../config.h"
#include LATTICE_PROPERTIES

#define CYLINDER
#define GPU_INDEX 0

#define X_PERIODIC 0 // or 0
#define Y_PERIODIC 0 // or 0
#define Z_PERIODIC 1 // or 0

constexpr bool rotated_coordinates = true;
constexpr bool triangular = true;

constexpr MassBC MASS_CONSERV = MassBC::Equilibrium;
constexpr MassBC BCF_MASS_CONSERV = MassBC::Equilibrium;

constexpr int BLOCK_SIZE = 16; // Maxmum block based on the register load

constexpr int D = 8;    // Diameter of the cylinder
constexpr int R = D / 2; // radius of the cylinder
constexpr real D_WALL = toReal(D);
constexpr real R_WALL = 0.5 * D_WALL;

constexpr int LW = 8 * D; // inlet from cylinder
constexpr int LE = 24 * D; // outlet from cylinder
constexpr int LN = 5 * D; // top wall from cylinder (y-dir)
constexpr int LS = LN;    // bottom wall from cylinder

constexpr int NX = LW + D + LE; // size x of the grid
constexpr int NY = LN + D + LS; // size y of the grid
constexpr int NZ = 5 * D;       // size z of the grid in one GPU

constexpr real XC = LW + 0.5 * (D - 1); // Center of the cylinder Xc
constexpr real YC = LS + 0.5 * (D - 1); // Center of the cylinder yc

constexpr real RE = 100;      // Reynolds number
constexpr real U_MAX = 0.1;   // lattice characteristic velocity
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

// ---------- RUNTIME CONSTANTS (HOST) ----------
inline real h_Hxx[Q] = {0};
inline real h_Hyy[Q] = {0};
inline real h_Hzz[Q] = {0};
inline real h_Hxy[Q] = {0};
inline real h_Hxz[Q] = {0};
inline real h_Hyz[Q] = {0};

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

extern __constant__ int d_NB;
extern __constant__ int d_NBCF;

#endif // CONSTANTS_H