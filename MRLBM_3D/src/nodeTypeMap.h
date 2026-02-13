#ifndef NODETYPEMAP_H
#define NODETYPEMAP_H

    /*   N            
    +----+----+
    | 64 |128 |      
 W  +----+----+  E      (BACK, z=0)
    | 16 | 32 |
    +----+----+
         S                 
/*
/*       N       
    +----+----+
    |  4 |  8 |      
 W  +----+----+  E      (FRONT, z=NZ)
    |  1 |  2 |
    +----+----+
         S
*/

// --- SPECIAL ---
constexpr nodeType_t SOLID_NODE = 0;
constexpr nodeType_t BULK = 255;

// --- FACE ---
constexpr nodeType_t NORTH = 51;
constexpr nodeType_t SOUTH = 204;
constexpr nodeType_t WEST = 170;
constexpr nodeType_t EAST = 85;
constexpr nodeType_t FRONT = 15;
constexpr nodeType_t BACK = 240;

// --- EDGE ---
constexpr nodeType_t NORTH_WEST = 34;
constexpr nodeType_t SOUTH_WEST = 136;
constexpr nodeType_t WEST_FRONT = 10;
constexpr nodeType_t WEST_BACK = 160;

constexpr nodeType_t NORTH_EAST = 17;
constexpr nodeType_t SOUTH_EAST = 68;
constexpr nodeType_t EAST_FRONT = 5;
constexpr nodeType_t EAST_BACK = 80;

constexpr nodeType_t NORTH_FRONT = 3;
constexpr nodeType_t NORTH_BACK = 48;

constexpr nodeType_t SOUTH_FRONT = 12;
constexpr nodeType_t SOUTH_BACK = 192;

// --- CORNER ---
constexpr nodeType_t NORTH_WEST_FRONT = 2;
constexpr nodeType_t NORTH_WEST_BACK = 32;
constexpr nodeType_t SOUTH_WEST_FRONT = 8;
constexpr nodeType_t SOUTH_WEST_BACK = 128;

constexpr nodeType_t NORTH_EAST_FRONT = 1;
constexpr nodeType_t NORTH_EAST_BACK = 16;
constexpr nodeType_t SOUTH_EAST_FRONT = 4;
constexpr nodeType_t SOUTH_EAST_BACK = 64;

// Curved boundary
constexpr nodeType_t INNER_NODE = 10000;
constexpr nodeType_t BCFLUID_NODE = 1000;

#define MISSING_DEFINITION (0b11111111111111111111111111111111)

#endif