#pragma once

#include <crts.h>
#include "matrix_gen_slave_param.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void SLAVE_FUN(preProcessY_gradY_partition_naive)(preProcessY_gradY_partition_param_t* para_p);

extern void SLAVE_FUN(preProcessY_gradY_div_meshV_naive)(preProcessY_gradY_div_meshV_param_t* para_p);


#ifdef __cplusplus
}
#endif