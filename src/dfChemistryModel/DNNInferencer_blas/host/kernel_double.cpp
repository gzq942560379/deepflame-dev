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