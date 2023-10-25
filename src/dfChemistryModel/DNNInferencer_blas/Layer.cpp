#include "Layer.H"
#include <cmath>
// #include <cblas.h>
#include <sstream>
#include <fstream>
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <crts.h>
#include "slave_kernel.h"
    
#ifdef __cplusplus
extern "C" {
#endif

extern void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);

#ifdef __cplusplus
}
#endif

#ifdef DEF_PROFILING

template<typename DataType>
void Linear<DataType>::profiling_reset(){
    sample_processed_ = 0;
    infer_time_ = 0.;
    infer_gemm_time_ = 0.;
    infer_add_time_ = 0.;
}

template<typename DataType>
void LinearGELU<DataType>::profiling_reset(){
    sample_processed_ = 0;
    infer_time_ = 0.;
    infer_gemm_time_ = 0.;
    infer_add_time_ = 0.;
    infer_gelu_time_ = 0.;
}

// print_profiling_info
template<typename DataType>
void Linear<DataType>::print_profiling_info(){
    int mpirank;
    int mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    if(mpirank == 0){
        std::cout << "Linear profiling info : " << std::endl;
        std::cout << "sample_processed : " << sample_processed_ << std::endl;
        std::cout << "infer_time : " << infer_time_ << std::endl;
        std::cout << "infer_gemm_time : " << infer_gemm_time_ << ", " << infer_gemm_time_ * 100. /  infer_time_ << "%" << std::endl;
        std::cout << "infer_add_time : " << infer_add_time_ << ", " << infer_add_time_ * 100. /  infer_time_ << "%" << std::endl;
        double infer_other_time = infer_time_ - infer_gemm_time_ - infer_add_time_;
        std::cout << "infer_other_time : " << infer_other_time << ", " << infer_other_time * 100. /  infer_time_ << "%" << std::endl;
    }
}

// print_profiling_info
template<typename DataType>
void LinearGELU<DataType>::print_profiling_info(){
    int mpirank;
    int mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    if(mpirank == 0){
        std::cout << "LinearGELU profiling info : " << std::endl;
        std::cout << "sample_processed : " << sample_processed_ << std::endl;
        std::cout << "infer_time : " << infer_time_ << std::endl;
        std::cout << "infer_gemm_time : " << infer_gemm_time_ << ", " << infer_gemm_time_ * 100. /  infer_time_ << "%" << std::endl;
        std::cout << "infer_add_time : " << infer_add_time_ << ", " << infer_add_time_ * 100. /  infer_time_ << "%" << std::endl;
        std::cout << "infer_gelu_time : " << infer_gelu_time_ << ", " << infer_gelu_time_ * 100. /  infer_time_ << "%" << std::endl;
        double infer_other_time = infer_time_ - infer_gemm_time_ - infer_add_time_ - infer_gelu_time_;
        std::cout << "infer_other_time : " << infer_other_time << ", " << infer_other_time * 100. /  infer_time_ << "%" << std::endl;
    }
}

#endif


template<>
void Linear<float>::load_parameters(const std::string& dir, int64_t layer_id){
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(!flag_mpi_init){
        std::cerr << "DNNInferencer_blas::load_models : MPI is not initialized" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int mpirank;
    int mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    if(mpirank == 0){
        std::stringstream ss1,ss2;
        ss1 << dir << "/" << "linear_" << layer_id << "_weights_rowmajor_" << in_features_ << "_" << out_features_ << ".data";
        std::string weights_path = ss1.str();
        ss2 << dir << "/" << "linear_" << layer_id << "_bias_" << out_features_ << ".data";
        std::string bias_path = ss2.str();
        // weights
        std::ifstream weights_file(weights_path, std::ios::binary);
        if(!weights_file.is_open()){
            std::cerr << "open weights file error : " << weights_path << std::endl << std::flush;
            abort();
        }
        weights_file.read(reinterpret_cast<char*>(weights_.data()), weights_.bytes_num());
        weights_file.close();
        // bias
        std::ifstream bias_file(bias_path, std::ios::binary);
        if(!bias_file.is_open()){
            std::cerr << "open bias file error : " << bias_path << std::endl << std::flush;
            abort();
        }
        bias_file.read(reinterpret_cast<char*>(bias_.data()), bias_.bytes_num());
        bias_file.close();
    }

    MPI_Bcast(weights_.data(), weights_.element_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_.data(), bias_.element_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

template<>
void LinearGELU<float>::load_parameters(const std::string& dir, int64_t layer_id){
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(!flag_mpi_init){
        std::cerr << "DNNInferencer_blas::load_models : MPI is not initialized" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int mpirank;
    int mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    if(mpirank == 0){
        std::stringstream ss1,ss2;
        ss1 << dir << "/" << "linear_" << layer_id << "_weights_rowmajor_" << in_features_ << "_" << out_features_ << ".data";
        std::string weights_path = ss1.str();
        ss2 << dir << "/" << "linear_" << layer_id << "_bias_" << out_features_ << ".data";
        std::string bias_path = ss2.str();
        
        // weights
        std::ifstream weights_file(weights_path, std::ios::binary);
        if(!weights_file.is_open()){
            std::cerr << "open weights file error : " << weights_path << std::endl << std::flush;
            abort();
        }
        weights_file.read(reinterpret_cast<char*>(weights_.data()), weights_.bytes_num());
        weights_file.close();

        // bias
        std::ifstream bias_file(bias_path, std::ios::binary);
        if(!bias_file.is_open()){
            std::cerr << "open bias file error : " << bias_path << std::endl << std::flush;
            abort();
        }
        bias_file.read(reinterpret_cast<char*>(bias_.data()), bias_.bytes_num());
        bias_file.close();
    }

    MPI_Bcast(weights_.data(), weights_.element_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_.data(), bias_.element_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void gelu_navie(int64_t len, float* data){
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        data[i] = 0.5f * x * (1.f + tanhf(sqrtf(2.f / M_PI) * (x + 0.044715f * powf(x, 3.f))));
    }
}

void gelu_navie(int64_t len, double* data){
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        double x = data[i];
        data[i] = 0.5 * x * (1. + tanh(sqrt(2. / M_PI) * (x + 0.044715 * pow(x, 3.))));
    }
}


inline float tanh_exp(float x){
    return 1.f - 2.f / (expf(2.f * x) + 1.f);
}

inline double tanh_exp(double x){
    return 1. - 2. / (exp(2. * x) + 1.);
}

inline float tanh_exp_8(float x){
    if(x > 8.f) return 1.;
    if(x < -8.f) return -1.;
    return 1.f - 2.f / (expf(2.f * x) + 1.f);
}

inline double tanh_exp_8(double x){
    if(x > 8.) return 1.;
    if(x < -8.) return -1.;
    return 1. - 2. / (exp(2. * x) + 1.);
}

void gelu_exp(int64_t len, float* data){
    const float const_1 = sqrtf(2.f / M_PI);
    const float const_2 = 0.044715f;
    const float one = 1.f;
    const float half = 0.5;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        float x = data[i];
        data[i] = half * x * (one + tanh_exp(const_1 * (x + const_2 * x * x * x)));
    }
}

void gelu_exp(int64_t len, double* data){
    const double const_1 = sqrt(2. / M_PI);
    const double const_2 = 0.044715;
    const double one = 1.;
    const double half = 0.;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        double x = data[i];
        data[i] = half * x * (one + tanh_exp(const_1 * (x + const_2 * x * x * x)));
    }
}

void gelud_exp(int64_t len, float* data){
    const double const_1 = sqrtf(2. / M_PI);
    const double const_2 = 0.044715;
    const double one = 1.;
    const double half = 0.5;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int64_t i = 0; i < len; ++i){
        double x = data[i];
        data[i] = half * x * (one + tanh_exp(const_1 * (x + const_2 * x * x * x)));
    }
}

// TODO 
void gelu_naive_slave(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_naive)), &para);
    CRTS_athread_join();
}

void gelu_exp_slave(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_exp)), &para);
    CRTS_athread_join();
}

void gelu_ldm_slave(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_ldm)), &para);
    CRTS_athread_join();
}

void gelu_ldm_lookup_slave(int64_t len, float* data){
    gelu_s_param_t para;
    para.len = len;
    para.data = data;
    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_ldm_lookup)), &para);
    CRTS_athread_join();
}

void bias_naive(Tensor<float>& input, const Tensor<float>& bias){
    int64_t row = input.dim(0);
    int64_t col = input.dim(1);
    int64_t ld = col;
    float* input_data = input.data();
    const float* bias_data = bias.data();
    for(int64_t r = 0; r < row; ++r){
        for(int64_t c = 0; c < col; ++c){
            input_data[r * ld + c] += bias_data[c];
        }
    }
}

void bias_naive_slave(Tensor<float>& input, const Tensor<float>& bias){

    bias_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_s_naive)), &para);
    CRTS_athread_join();
}

void bias_ldm_slave(Tensor<float>& input, const Tensor<float>& bias){
    bias_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_s_ldm)), &para);
    CRTS_athread_join();
}

void bias_gelu_ldm_lookup_slave(Tensor<float>& input, const Tensor<float>& bias){
    bias_gelu_s_param_t para;

    para.row = input.dim(0);
    para.col = input.dim(1);
    para.input = input.data();
    para.bias = const_cast<float*>(bias.data());

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(bias_gelu_s_ldm_lookup)), &para);
    CRTS_athread_join();
}

template<>
void Linear<float>::forward(const Tensor<float>& input, Tensor<float>& output){
#ifdef DEF_PROFILING
    double infer_start = MPI_Wtime();
#endif

    char transA = 'N';
    char transB = 'N';
    float alpha = 1.f;
    float beta = 0.f;
    int m = out_features_;
    int n = input.dim(0);
    int k = in_features_;
    float* A = weights_.data();
    int lda = out_features_;
    float* B = const_cast<float*>(input.data());
    int ldb = input.dim(1);
    float*  C = output.data();
    int ldc = output.dim(1);

#ifdef DEF_PROFILING
    double infer_gemm_start = MPI_Wtime();
#endif

    sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

#ifdef DEF_PROFILING
    double infer_gemm_end = MPI_Wtime();
    double infer_add_start = MPI_Wtime();
#endif

    // bias_naive(output, bias_);
    // bias_naive_slave(output, bias_);
    bias_ldm_slave(output, bias_);

#ifdef DEF_PROFILING
    double infer_add_end = MPI_Wtime();
    double infer_end = MPI_Wtime();
    sample_processed_ += input.dim(0);
    infer_gemm_time_ += infer_gemm_end - infer_gemm_start;
    infer_add_time_ += infer_add_end - infer_add_start;
    infer_time_ += infer_end - infer_start;
#endif
}

template<>
void LinearGELU<float>::forward(const Tensor<float>& input, Tensor<float>& output){
#ifdef DEF_PROFILING
    double infer_start = MPI_Wtime();
#endif

    char transA = 'N';
    char transB = 'N';
    float alpha = 1.f;
    float beta = 0.f;
    int m = out_features_;
    int n = input.dim(0);
    int k = in_features_;
    float* A = weights_.data();
    int lda = out_features_;
    float* B = const_cast<float*>(input.data());
    int ldb = input.dim(1);
    float*  C = output.data();
    int ldc = output.dim(1);

#ifdef DEF_PROFILING
    double infer_gemm_start = MPI_Wtime();
#endif
    sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

#ifdef DEF_PROFILING
    double infer_gemm_end = MPI_Wtime();
    double infer_add_start = MPI_Wtime();
#endif

    // bias_naive(output, bias_);
    // bias_naive_slave(output, bias_);
    // bias_ldm_slave(output, bias_);

#ifdef DEF_PROFILING
    double infer_add_end = MPI_Wtime();
    double infer_gelu_start = MPI_Wtime();
#endif

    // GELU
    // gelu_navie(output.element_num(), output.data());
    // gelu_exp(output.element_num(), output.data());
    // gelu_naive_slave(output.element_num(), output.data());
    // gelu_exp_slave(output.element_num(), output.data());
    // gelu_ldm_slave(output.element_num(), output.data());
    // gelu_ldm_lookup_slave(output.element_num(), output.data());

    bias_gelu_ldm_lookup_slave(output,bias_);

#ifdef DEF_PROFILING
    double infer_gelu_end = MPI_Wtime();
    double infer_end = MPI_Wtime();
    sample_processed_ += input.dim(0);
    infer_gemm_time_ += infer_gemm_end - infer_gemm_start;
    infer_add_time_ += infer_add_end - infer_add_start;
    infer_gelu_time_ += infer_gelu_end - infer_gelu_start;
    infer_time_ += infer_end - infer_start;
#endif
}

template class Tensor<float>;
template class Linear<float>;
template class LinearGELU<float>;