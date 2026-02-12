#ifndef NODETYPEMAP_H
#define NODETYPEMAP_H

/*
+----+----+
|  4 |  8 |
+----+----+
|  1 |  2 |
+----+----+
*/

constexpr nodeType_t BULK = 15;
constexpr nodeType_t SOLID = 0;

// Edges
constexpr nodeType_t NORTH = 3;
constexpr nodeType_t SOUTH = 12;
constexpr nodeType_t EAST = 5;
constexpr nodeType_t WEST = 10;

// Corners
constexpr nodeType_t NORTH_EAST = 1;
constexpr nodeType_t NORTH_WEST = 2;
constexpr nodeType_t SOUTH_EAST = 4;
constexpr nodeType_t SOUTH_WEST = 8;

// Curved boundary
constexpr nodeType_t INNER_NODE = 10000;
constexpr nodeType_t BCFLUID_NODE = 100;

#define MISSING_DEFINITION (0b11111111111111111111111111111111)

#endif