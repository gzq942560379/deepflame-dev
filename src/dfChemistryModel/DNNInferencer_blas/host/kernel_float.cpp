#include "kernel.H"
#include <cmath>
#include <sstream>
#include <fstream>
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <crts.h>
#include "slave_kernel.h"

template<>
void gelu_naive_slave<float>(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_naive)), &para);
    CRTS_athread_join();
}

template<>
void gelu_exp_slave<float>(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_exp)), &para);
    CRTS_athread_join();
}

template<>
void gelu_ldm_slave<float>(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_ldm)), &para);
    CRTS_athread_join();
}

template<>
void gelu_ldm_lookup_slave<float>(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_ldm_lookup)), &para);
    CRTS_athread_join();
}

template<>
void gelu_fastexp<float>(int64_t len, float* data){
    const float const_sqrt_2_div_pi = static_cast<float>(0.7978845608028654);
    const float const_2 = static_cast<float>(0.044715);
    const float const_half = static_cast<float>(0.5);
    const float const_max = 8;
    const float const_one = 1.;
    const float const_two = 2.;
    const uint32_t const_sign_mask = 0x80000000;
    const uint32_t const_abs_value_mask = 0x7FFFFFFF;
    const float const_log2e = 1.442695040;
    const float exp_coef[2] = { -0.05288671, 0.99232129};
    const uint32_t const_shift_float = 23;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        float tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint32_t tanh_x_int = *(uint32_t*)&tanh_x;
        uint32_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint32_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        float abs_tanh_x = *(float*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        float exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        float exp_xi = std::floor(exp_x);
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

template<>
void gelu_fastexp_slave<float>(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_fastexp)), &para);
    CRTS_athread_join();
}

template<>
void bias_naive_slave<float>(Tensor<float>& input, const Tensor<float>& bias){
    bias_s_param_t para;
    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_s_naive)), &para);
    CRTS_athread_join();
}

template<>
void bias_ldm_slave<float>(Tensor<float>& input, const Tensor<float>& bias){
    bias_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_s_ldm)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_ldm_lookup_slave<float>(Tensor<float>& input, const Tensor<float>& bias){
    bias_gelu_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_s_ldm_lookup)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_ldm_lookup_prefetch_slave<float>(Tensor<float>& input, const Tensor<float>& bias){
    bias_gelu_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_s_ldm_lookup_prefetch)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_fastexp_slave<float>(Tensor<float>& input, const Tensor<float>& bias){
    bias_gelu_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_s_fastexp)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_fastexp_simd_slave<float>(Tensor<float>& input, const Tensor<float>& bias){
    bias_gelu_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_s_fastexp_simd)), &para);
    CRTS_athread_join();
}

