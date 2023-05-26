#include "DNNInferencer_blas.H"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <omp.h>

DNNInferencer_blas::DNNInferencer_blas() {
    char* env_tmp = getenv("DNN_BATCH_SIZE");
    if(env_tmp == NULL){
        this->batch_size_ = 131072;
    }else{
        this->batch_size_ = std::atol(env_tmp);;
    }

    buffer_alloced_ = true;
    output_buffer_.emplace_back(std::vector<float>(batch_size_ * layers_[1]));
    output_buffer_.emplace_back(std::vector<float>(batch_size_ * layers_[2]));
    output_buffer_.emplace_back(std::vector<float>(batch_size_ * layers_[3]));
    output_buffer_.emplace_back(std::vector<float>(batch_size_ * layers_[4]));

    FLOPs_per_sample_ = 2. * (layers_[0] * layers_[1] + layers_[1] * layers_[2] + layers_[2] * layers_[3] + layers_[3] * layers_[4]);
}

DNNInferencer_blas::~DNNInferencer_blas() {
    for(int i = 0;i < model0_.size(); ++i){
        delete model0_[i];
    }
    model0_.clear();
    for(int i = 0;i < model1_.size(); ++i){
        delete model1_[i];
    }
    model1_.clear();
    for(int i = 0;i < model2_.size(); ++i){
        delete model2_[i];
    }
    model2_.clear();
}


void DNNInferencer_blas::load_models(const std::string dir){
    // init model
    model0_.push_back(new LinearGELU<float>(layers_[0],layers_[1]));
    model0_.push_back(new LinearGELU<float>(layers_[1],layers_[2]));
    model0_.push_back(new LinearGELU<float>(layers_[2],layers_[3]));
    model0_.push_back(new Linear<float>(layers_[3],layers_[4]));

    model1_.push_back(new LinearGELU<float>(layers_[0],layers_[1]));
    model1_.push_back(new LinearGELU<float>(layers_[1],layers_[2]));
    model1_.push_back(new LinearGELU<float>(layers_[2],layers_[3]));
    model1_.push_back(new Linear<float>(layers_[3],layers_[4]));
    
    model2_.push_back(new LinearGELU<float>(layers_[0],layers_[1]));
    model2_.push_back(new LinearGELU<float>(layers_[1],layers_[2]));
    model2_.push_back(new LinearGELU<float>(layers_[2],layers_[3]));
    model2_.push_back(new Linear<float>(layers_[3],layers_[4]));
    // load parameters
    for(int i = 0; i < model0_.size(); ++i){
        model0_[i]->load_parameters(dir+"/0", i);
    }
    for(int i = 0; i < model1_.size(); ++i){
        model1_[i]->load_parameters(dir+"/1", i);
    }
    for(int i = 0; i < model2_.size(); ++i){
        model2_[i]->load_parameters(dir+"/2", i);
    }
}

void DNNInferencer_blas::Inference_multiDNNs(
    const std::vector<float>& input0, std::vector<double>& output0, int64_t input_count0,
    const std::vector<float>& input1, std::vector<double>& output1, int64_t input_count1,
    const std::vector<float>& input2, std::vector<double>& output2, int64_t input_count2
){
    
    double dnn_infer_start = MPI_Wtime();

    if(input_count0 > 0){
        output0.resize(input_count0 * output_dim());

        for(int64_t sample_start = 0; sample_start < input_count0; sample_start += batch_size_){
            int64_t sample_end = std::min(input_count0, sample_start + batch_size_);
            int64_t sample_len = sample_end - sample_start;

            Tensor<float> input_tensor_0({sample_len, layers_[0]}, const_cast<float*>(input0.data()) + sample_start * input_dim());
            Tensor<float> input_tensor_1({sample_len, layers_[1]}, output_buffer_[0].data());
            Tensor<float> input_tensor_2({sample_len, layers_[2]}, output_buffer_[1].data());
            Tensor<float> input_tensor_3({sample_len, layers_[3]}, output_buffer_[2].data());
            Tensor<float> input_tensor_4({sample_len, layers_[4]}, output_buffer_[3].data());

            model0_[0]->forward(input_tensor_0, input_tensor_1);
            model0_[1]->forward(input_tensor_1, input_tensor_2);
            model0_[2]->forward(input_tensor_2, input_tensor_3);
            model0_[3]->forward(input_tensor_3, input_tensor_4);

            double* __restrict__ output0_ptr = output0.data() + sample_start * output_dim();
            const float* const __restrict__ input_tensor_4_ptr = input_tensor_4.data();

            for(int i = 0; i < input_tensor_4.element_num(); ++i){
                output0_ptr[i] = input_tensor_4_ptr[i];
            }
        }
    }

    if(input_count1 > 0){
        output1.resize(input_count1 * output_dim());

        for(int64_t sample_start = 0; sample_start < input_count1; sample_start += batch_size_){
            int64_t sample_end = std::min(input_count1, sample_start + batch_size_);
            int64_t sample_len = sample_end - sample_start;

            Tensor<float> input_tensor_0({sample_len, layers_[0]}, const_cast<float*>(input1.data()) + sample_start * input_dim());
            Tensor<float> input_tensor_1({sample_len, layers_[1]}, output_buffer_[0].data());
            Tensor<float> input_tensor_2({sample_len, layers_[2]}, output_buffer_[1].data());
            Tensor<float> input_tensor_3({sample_len, layers_[3]}, output_buffer_[2].data());
            Tensor<float> input_tensor_4({sample_len, layers_[4]}, output_buffer_[3].data());

            model1_[0]->forward(input_tensor_0, input_tensor_1);
            model1_[1]->forward(input_tensor_1, input_tensor_2);
            model1_[2]->forward(input_tensor_2, input_tensor_3);
            model1_[3]->forward(input_tensor_3, input_tensor_4);

            double* __restrict__ output1_ptr = output1.data() + sample_start * output_dim();
            const float* const __restrict__ input_tensor_4_ptr = input_tensor_4.data();

            for(int i = 0; i < input_tensor_4.element_num(); ++i){
                output1_ptr[i] = input_tensor_4_ptr[i];
            }
        }
    }

    if(input_count2 > 0){
        output2.resize(input_count2 * output_dim());

        for(int64_t sample_start = 0; sample_start < input_count2; sample_start += batch_size_){
            int64_t sample_end = std::min(input_count2, sample_start + batch_size_);
            int64_t sample_len = sample_end - sample_start;

            Tensor<float> input_tensor_0({sample_len, layers_[0]}, const_cast<float*>(input2.data()) + sample_start * input_dim());
            Tensor<float> input_tensor_1({sample_len, layers_[1]}, output_buffer_[0].data());
            Tensor<float> input_tensor_2({sample_len, layers_[2]}, output_buffer_[1].data());
            Tensor<float> input_tensor_3({sample_len, layers_[3]}, output_buffer_[2].data());
            Tensor<float> input_tensor_4({sample_len, layers_[4]}, output_buffer_[3].data());

            model2_[0]->forward(input_tensor_0, input_tensor_1);
            model2_[1]->forward(input_tensor_1, input_tensor_2);
            model2_[2]->forward(input_tensor_2, input_tensor_3);
            model2_[3]->forward(input_tensor_3, input_tensor_4);

            double* __restrict__ output2_ptr = output2.data() + sample_start * output_dim();
            const float* const __restrict__ input_tensor_4_ptr = input_tensor_4.data();

            for(int i = 0; i < input_tensor_4.element_num(); ++i){
                output2_ptr[i] = input_tensor_4_ptr[i];
            }
        }
    }

    double dnn_infer_end = MPI_Wtime();
    double dnn_infer_time = dnn_infer_end - dnn_infer_start;
    double FLOPs = (input_count0 + input_count1 + input_count2) * FLOPs_per_sample_;
    int num_threads = omp_get_max_threads();
    double theoretical_peak = 3.3792 / 48. * 2. * num_threads;
    double FLOPS = FLOPs / dnn_infer_time;
    double TFLOPS = FLOPS * 1e-12;
    double peak = TFLOPS * 100. / theoretical_peak;

    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

    if(mpirank == 0){
        std::cout << "Inference Performance ---------------" << std::endl;
        std::cout << "samples : " << (input_count0 + input_count1 + input_count2) << std::endl;
        std::cout << "batch size : " << batch_size_ << std::endl;
        std::cout << "Time : " << dnn_infer_time << std::endl;
        std::cout << "FLOPS : " << FLOPs << std::endl;
        std::cout << "TFLOPS : " << TFLOPS << std::endl;
        std::cout << "Peak : " << peak << std::endl;
        std::cout << "-------------------------------------" << std::endl;
    } 
}



