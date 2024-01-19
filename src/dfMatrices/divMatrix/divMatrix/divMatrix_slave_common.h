#pragma once

#include <stdint.h>

typedef double scalar;
typedef int64_t label;

#define DIV_BLOCK_MASK ((1 << (row_block_bit_)) - 1)
#define DIV_BLOCK_MASK_INVERSE (~DIV_BLOCK_MASK)

#define DIV_BLOCK_INDEX(row) ((row) >> (row_block_bit_))
#define DIV_BLOCK_ROW(row) ((row) & DIV_BLOCK_MASK)
#define DIV_BLOCK_START(bi) (bi << row_block_bit_)
// #define DIV_BLOCK_END(rbs) (std::min((rbs) + (row_block_size_), (row_)))
#define DIV_BLOCK_END(rbs) ((rbs) + (row_block_size_))
// #define DIV_BLOCK_LEN(rbs,rbe) ((rbe - rbs))
#define DIV_BLOCK_LEN(rbs,rbe) (row_block_size_)
#define DIV_BLOCK_TAIL DIV_BLOCK_ROW(row_)
#define DIV_BLOCK_COUNT (((row_) + (row_block_size_) - 1) >> (row_block_bit_))

#define DIV_INDEX_BLOCK_START(rbs) ((rbs) * (distance_count_))
#define DIV_COL_OFFSET(divcol) ((divcol) << (row_block_bit_))
#define DIV_INDEX(row,divcol) (((row) & (DIV_BLOCK_MASK_INVERSE)) * (distance_count_) + DIV_COL_OFFSET(divcol) + DIV_BLOCK_ROW(row))


#define slave_min(x,y) ((x)<(y)?(x):(y))
#define slave_max(x,y) ((x)<(y)?(y):(x))