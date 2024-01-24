#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"

void gelu_s_fastexp(gelu_s_param_t *para_p){
    const float const_sqrt_2_div_pi = 0.7978845608028654;
    const float const_2 = 0.044715;
    const float const_half = 0.5;
    const float const_max = 8;
    const float const_one = 1.;
    const float const_two = 2.;
    const uint32_t const_sign_mask = 0x80000000;
    const uint32_t const_abs_value_mask = 0x7FFFFFFF;
    const float const_log2e = 1.442695040;
    const float exp_coef[2] = { -0.05288671, 0.99232129};
    const uint32_t const_shift_float = 23;

    int64_t len = para_p->len;
    float* data = para_p->data;

    int64_t start = CRTS_tid * len / CRTS_MAX_SPE_NUM;
    int64_t end = (CRTS_tid + 1) * len / CRTS_MAX_SPE_NUM;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = start; i < end; ++i){
        float x = data[i];
        float tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint32_t tanh_x_int = *(uint32_t*)&tanh_x;
        uint32_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint32_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        float abs_tanh_x = *(float*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        float exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        float exp_xi = floorf(exp_x);
        uint32_t exp_xi_int = exp_xi;
        float exp_xf = exp_x - exp_xi;
        float exp_k = exp_coef[1] * exp_xf + exp_coef[0] + const_one;
        uint32_t exp_e = *(uint32_t*)&exp_k;
        exp_e += (exp_xi_int << const_shift_float);
        float exp_ret = *(float*)&exp_e;
        float abs_ret = const_one - const_two / (exp_ret + const_one);
        uint32_t tanh_ret_int = *(uint32_t*)&abs_ret | tanh_x_sign;
        float tanh_ret = *(float*)&tanh_ret_int;
        data[i] = const_half * x * (const_one + tanh_ret);
    }
}