#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"

void gelu_d_fastexp(gelu_d_param_t *para_p){
    const double const_sqrt_2_div_pi = 0.7978845608028654;
    const double const_2 = 0.044715;
    const double const_half = 0.5;
    const double const_max = 8;
    const double const_one = 1.;
    const double const_two = 2.;
    const uint64_t const_sign_mask = 0x8000000000000000;
    const uint64_t const_abs_value_mask = 0x7FFFFFFFFFFFFFFF;
    const double const_log2e = 1.442695040;
    const double exp_coef[2] = { -0.05288671, 0.99232129};
    const uint64_t const_shift_double = 52;

    int64_t len = para_p->len;
    double* data = para_p->data;

    int64_t start = CRTS_tid * len / CRTS_MAX_SPE_NUM;
    int64_t end = (CRTS_tid + 1) * len / CRTS_MAX_SPE_NUM;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = start; i < end; ++i){
        double x = data[i];
        double tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint64_t tanh_x_int = *(uint64_t*)&tanh_x;
        uint64_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint64_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        double abs_tanh_x = *(double*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        double exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        double exp_xi = floor(exp_x);
        uint64_t exp_xi_int = exp_xi;
        double exp_xf = exp_x - exp_xi;
        double exp_k = exp_coef[1] * exp_xf + exp_coef[0] + const_one;
        uint64_t exp_e = *(uint64_t*)&exp_k;
        exp_e += (exp_xi_int << const_shift_double);
        double exp_ret = *(double*)&exp_e;
        double abs_ret = const_one - const_two / (exp_ret + const_one);
        uint64_t tanh_ret_int = *(uint64_t*)&abs_ret | tanh_x_sign;
        double tanh_ret = *(double*)&tanh_ret_int;
        data[i] = const_half * x * (const_one + tanh_ret);
    }
}