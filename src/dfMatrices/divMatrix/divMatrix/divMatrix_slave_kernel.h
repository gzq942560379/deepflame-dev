#pragma once

#include <crts.h>
#include "divMatrix_slave_param.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void SLAVE_FUN(divMatrix_SpMV_naive)(divMatrix_SpMV_param_t* para_p);

extern void SLAVE_FUN(divMatrix_sumA_naive)(divMatrix_sumA_param_t* para_p);

extern void SLAVE_FUN(divMatrix_Jacobi_naive)(divMatrix_Jacobi_param_t* para_p);

extern void SLAVE_FUN(divMatrix_value_transfer_navie)(divMatrix_value_transfer_param_t* para_p);

#ifdef __cplusplus
}
#endif