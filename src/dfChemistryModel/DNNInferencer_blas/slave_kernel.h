#pragma once

#include <crts.h>
#include "slave_param.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void SLAVE_FUN(gelu_s_naive)(gelu_s_param_t* para_p);
extern void SLAVE_FUN(gelu_s_exp)(gelu_s_param_t* para_p);
extern void SLAVE_FUN(gelu_s_ldm)(gelu_s_param_t* para_p);
extern void SLAVE_FUN(gelu_s_ldm_lookup)(gelu_s_param_t* para_p);


extern void SLAVE_FUN(bias_s_naive)(bias_s_param_t* para_p);
extern void SLAVE_FUN(bias_s_ldm)(bias_s_param_t* para_p);


extern void SLAVE_FUN(bias_gelu_s_ldm_lookup)(bias_gelu_s_param_t* para_p);

#ifdef __cplusplus
}
#endif