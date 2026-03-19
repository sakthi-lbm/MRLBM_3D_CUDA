#pragma once

#include "../../common/globalStructs.cuh"

constexpr size_t NUMBER_OF_MOMENTS = 10;

constexpr int Q = 27; // lattice directions
constexpr int QF = 9; // lattice directions for halo interface
constexpr real W0 = 8.0 / 27.0;
constexpr real W1 = 2.0 / 27.0;
constexpr real W2 = 1.0 / 54.0;
constexpr real W3 = 1.0 / 216.0;

constexpr real h_w[Q] = {W0,
                         W1, W1, W1, W1, W1, W1,
                         W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,
                         W3, W3, W3, W3, W3, W3, W3, W3};

constexpr int h_cx[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
constexpr int h_cy[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
constexpr int h_cz[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};

constexpr int opp[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25};

constexpr real as2 = 3.0;
constexpr real cs2 = 1.0 / as2;

constexpr real F_M_0_SCALE = 1.0;
constexpr real F_M_I_SCALE = as2;
constexpr real F_M_II_SCALE = 0.5 * as2 * as2;
constexpr real F_M_IJ_SCALE = as2 * as2;
