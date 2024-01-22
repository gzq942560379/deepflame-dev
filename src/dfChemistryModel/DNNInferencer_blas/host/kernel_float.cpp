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
