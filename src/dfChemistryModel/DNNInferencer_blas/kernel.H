#pragma once

#include <cmath>
#include <cblas.h>
#include <iostream>
#include "Tensor.H"

template<typename T>
static T tanh_exp(T x);

template<typename T>
static T fast_exp(T x);

template<typename T>
void gelu_navie(int64_t len, T* data);

template<typename T>
void gelu_exp(int64_t len, T* data);

template<typename T>
void gelu_exp_fusion(int64_t len, T* data);

#ifdef __ARM_FEATURE_SVE
template<typename T>
void gelu_exp_simd(int64_t len, T* data);
#endif

template<typename T>
void gelu_lookup(int64_t len, T* data);

template<typename T>
void bias_gelu_exp_fusion(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_gelu_lookup_fusion(Tensor<T>& input, const Tensor<T>& bias);

#ifdef __cplusplus
extern "C"{
#endif

extern void sgemm_(char* transA, char* transB, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float* beta, float* c, int* ldc);
extern void dgemm_(char* transA, char* transB, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double* beta, double* c, int* ldc);
// extern void zgemm_(char* transA, char* transB, int* m, int* n, int* k, std::complex<double>* alpha, const std::complex<double>* a, int* lda, const std::complex<double>* b, int* ldb, std::complex<double>* beta, std::complex<double>* c, int* ldc);

#ifdef __cplusplus
}
#endif

template<typename T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta, T *c, int ldc);

// implement ----------------------------------------------------------------------------

template<typename T>
void bias_naive(Tensor<T>& input, const Tensor<T>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    T* input_data = input.data();
    const T* bias_data = bias.data();
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t r = 0; r < row; ++r){
        for(int64_t c = 0; c < col; ++c){
            input_data[r * ld + c] += bias_data[c];
        }
    }
}

#ifdef __ARM_FEATURE_SVE
template<typename T>
void gelu_exp_simd(int64_t len, T* data){
    return gelu_exp_fusion(len, data);
}
#endif


