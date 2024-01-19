#pragma once

#include <crts.h>
#include "divMatrix_slave_param.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void SLAVE_FUN(divMatrix_SpMV_naive)(divMatrix_SpMV_param_t* para_p);

#ifdef __cplusplus
}
#endif