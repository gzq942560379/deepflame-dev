#pragma once

#include <cmath>
#include <iostream>
#include "Tensor.H"

#ifdef __cplusplus
extern "C" {
#endif

extern void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);

#ifdef __cplusplus
}
#endif

inline void gemm(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc){
    sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc){
    dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gelu_navie(int64_t len, float* data);
void gelu_navie(int64_t len, double* data);

template<typename T>
T tanh_exp(T x);

template<typename T>
T tanh_exp_8(T x);

template<typename T>
void gelu_exp(int64_t len, T* data);

template<typename T>
void gelu_naive_slave(int64_t len, T* data);

template<typename T>
void gelu_exp_slave(int64_t len, T* data);

template<typename T>
void gelu_ldm_slave(int64_t len, T* data);

template<typename T>
void gelu_ldm_lookup_slave(int64_t len, T* data);

template<typename T>
void gelu_fastexp(int64_t len, T* data);

template<typename T>
void gelu_fastexp_slave(int64_t len, T* data);

template<typename T>
void bias_naive(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_naive_slave(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_ldm_slave(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_gelu_ldm_lookup_slave(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_gelu_ldm_lookup_prefetch_slave(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_gelu_fastexp_slave(Tensor<T>& input, const Tensor<T>& bias);

template<typename T>
void bias_gelu_fastexp_simd_slave(Tensor<T>& input, const Tensor<T>& bias);

// impletement -----------------------------------------------------------

template<typename T>
T tanh_exp(T x){
    const T one = 1.;
    const T two = 2.;
    return one - two / (std::exp(two * x) + one);
}

template<typename T>
T tanh_exp_8(T x){
    const T one = 1.;
    const T neg_one = 1.;
    const T two = 2.;
    const T eight = 8.;
    const T neg_eight = -8;
    if(x > eight) return one;
    if(x < neg_eight) return neg_one;
    return one - two / (std::exp(two * x) + one);
}

template<typename T>
void gelu_navie(int64_t len, T* data){
    const T half = 0.5;
    const T one = 1.;
    const T two = 2.;
    const T pi = M_PI;
    const T const_0 = 0.044715;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        data[i] = half * x * (one + std::tanh(std::sqrt(two / pi) * (x + const_0 * x * x * x)));
    }
}

template<typename T>
void gelu_exp(int64_t len, T* data){
    const T const_1 = 0.7978845608028654;
    const T const_2 = 0.044715;
    const T one = 1.;
    const T half = 0.5;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        T x = data[i];
        data[i] = half * x * (one + tanh_exp<T>(const_1 * (x + const_2 * x * x * x)));
    }
}

template<typename T>
void bias_naive(Tensor<T>& input, const Tensor<T>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    T* input_data = input.data();
    const T* bias_data = bias.data();
    for(int64_t r = 0; r < row; ++r){
        for(int64_t c = 0; c < col; ++c){
            input_data[r * ld + c] += bias_data[c];
        }
    }
}