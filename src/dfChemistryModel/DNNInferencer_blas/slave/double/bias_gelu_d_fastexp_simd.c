#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"

#define MAX_BIAS 3200
#define MAX_BIAS_PADDING 8
#define MAX_BIAS_SIZE (MAX_BIAS + MAX_BIAS_PADDING)
#define LEN_BLOCK_SIZE 1024
#define SIMD_WIDTH 8

void bias_gelu_d_fastexp_simd(bias_gelu_d_param_t *para_p){
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

    double* input = para_p->input;
    double* bias = para_p->bias;
    int64_t row = para_p->row;
    int64_t col = para_p->col;

    int64_t len = row * col;

    int64_t local_start = CRTS_tid * len / CRTS_MAX_SPE_NUM;
    int64_t local_end = (CRTS_tid + 1) * len / CRTS_MAX_SPE_NUM;
    int64_t local_len = local_end - local_start;

    double bias_ldm[MAX_BIAS_SIZE];
    CRTS_dma_get(&bias_ldm, bias, col * sizeof(double));

    for(int64_t i = 0; i < MAX_BIAS_PADDING; ++i){
        bias_ldm[col + i] = bias_ldm[i];
    }

    double input_ldm[LEN_BLOCK_SIZE];

    for(int64_t bi = local_start; bi < local_end; bi += LEN_BLOCK_SIZE){
        int64_t iter_start = bi;
        int64_t iter_end = min(bi + LEN_BLOCK_SIZE, local_end);
        int64_t iter_len = iter_end - iter_start;
        CRTS_dma_get(input_ldm, input + iter_start, iter_len * sizeof(double));
        for(int64_t i = 0; i < iter_len; ++i){
            int64_t c = (iter_start + i) % col;
            double x = input_ldm[i] + bias_ldm[c];
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
            input_ldm[i] = const_half * x * (const_one + tanh_ret);
        }
        CRTS_dma_put(input + iter_start, input_ldm, iter_len * sizeof(double));
    }
}