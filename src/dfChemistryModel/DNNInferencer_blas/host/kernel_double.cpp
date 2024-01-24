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
void gelu_naive_slave<double>(int64_t len, double* data){
    gelu_d_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_d_naive)), &para);
    CRTS_athread_join();
}

template<>
void gelu_exp_slave<double>(int64_t len, double* data){
    gelu_d_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_d_exp)), &para);
    CRTS_athread_join();
}

template<>
void gelu_ldm_slave<double>(int64_t len, double* data){
    gelu_d_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_d_ldm)), &para);
    CRTS_athread_join();
}

template<>
void gelu_ldm_lookup_slave<double>(int64_t len, double* data){
    gelu_d_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_d_ldm_lookup)), &para);
    CRTS_athread_join();
}

template<>
void gelu_fastexp<double>(int64_t len, double* data){
    const double const_sqrt_2_div_pi = static_cast<double>(0.7978845608028654);
    const double const_2 = static_cast<double>(0.044715);
    const double const_half = static_cast<double>(0.5);
    const double const_max = 8;
    const double const_one = 1.;
    const double const_two = 2.;
    const uint64_t const_sign_mask = 0x8000000000000000;
    const uint64_t const_abs_value_mask = 0x7FFFFFFFFFFFFFFF;
    const double const_log2e = 1.442695040;
    const double exp_coef[2] = { -0.05288671, 0.99232129};
    const uint64_t const_shift_double = 52;

#ifdef _OPENMP    
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        double x = data[i];
        double tanh_x = const_sqrt_2_div_pi * (x + const_2 * x * x * x);
        uint64_t tanh_x_int = *(uint64_t*)&tanh_x;
        uint64_t tanh_x_sign = tanh_x_int & const_sign_mask;
        uint64_t abs_tanh_x_int = tanh_x_int & const_abs_value_mask;
        double abs_tanh_x = *(double*)&abs_tanh_x_int;
        abs_tanh_x = abs_tanh_x < const_max ? abs_tanh_x : const_max;
        double exp_x = const_two * abs_tanh_x;
        exp_x *= const_log2e;
        double exp_xi = std::floor(exp_x);
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

template<>
void gelu_fastexp_slave<double>(int64_t len, double* data){
    gelu_d_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_d_fastexp)), &para);
    CRTS_athread_join();
}

template<>
void bias_naive_slave<double>(Tensor<double>& input, const Tensor<double>& bias){
    bias_d_param_t para;
    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<double*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_d_naive)), &para);
    CRTS_athread_join();
}

template<>
void bias_ldm_slave<double>(Tensor<double>& input, const Tensor<double>& bias){
    bias_d_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<double*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_d_ldm)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_ldm_lookup_slave<double>(Tensor<double>& input, const Tensor<double>& bias){
    bias_gelu_d_param_t para;
    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<double*>(bias.data());
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_d_ldm_lookup)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_ldm_lookup_prefetch_slave<double>(Tensor<double>& input, const Tensor<double>& bias){
    bias_gelu_d_param_t para;
    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<double*>(bias.data());
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_d_ldm_lookup_prefetch)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_fastexp_slave<double>(Tensor<double>& input, const Tensor<double>& bias){
    bias_gelu_d_param_t para;
    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<double*>(bias.data());
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_d_fastexp)), &para);
    CRTS_athread_join();
}

template<>
void bias_gelu_fastexp_simd_slave<double>(Tensor<double>& input, const Tensor<double>& bias){
    bias_gelu_d_param_t para;
    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<double*>(bias.data());
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_d_fastexp_simd)), &para);
    CRTS_athread_join();
}