#pragma once
#include <crts.h>
#include "divMatrix_slave_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
    scalar* ApsiPtr;
    const scalar* psiPtr;
    const scalar* diagPtr;
    const scalar* off_diag_value_Ptr;
    const label* distance_list_;
    label row_block_bit_;
    label row_block_size_;
    label distance_count_;
    label block_count_;
    label row_;
} divMatrix_SpMV_param_t;

typedef struct{
    scalar* psiPtr;
    scalar* psiCopyPtr;
    const scalar* bPrimePtr;
    const scalar* diagPtr;
    const scalar* off_diag_value_Ptr;
    const label* distance_list_;
    label row_block_bit_;
    label row_block_size_;
    label distance_count_;
    label block_count_;
    label row_;
} divMatrix_Jacobi_param_t;

typedef struct{
    scalar* sumAPtr;
    const scalar* diagPtr;
    const scalar* off_diag_value_Ptr;
    label row_block_bit_;
    label row_block_size_;
    label distance_count_;
    label block_count_;
    label row_;
} divMatrix_sumA_param_t;

#ifdef __cplusplus
}
#endif