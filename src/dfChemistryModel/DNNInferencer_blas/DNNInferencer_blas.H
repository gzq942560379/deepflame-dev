#ifndef DNNInferencer_blas_H
#define DNNInferencer_blas_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string> 
#include <memory>
#include <cstring>
#include "Layer.H"

class DNNInferencer_blas
{
private:
    std::vector<int64_t> layers_{23,3200,1600,800,400,23};
    std::vector<Layer<float>*> model0_;
    std::vector<Layer<float>*> model1_;
    std::vector<Layer<float>*> model2_;

    double FLOPs_per_sample_;

    int64_t total_samples_;
    bool buffer_alloced_ = false;
    int64_t batch_size_;
    std::vector<std::vector<float>> output_buffer_;
public:

    int64_t input_dim() const {return layers_[0]; }
    int64_t output_dim() const {return layers_[5]; }

    int64_t batch_size() const {return batch_size_; }

    DNNInferencer_blas();
    ~DNNInferencer_blas();

    void load_models(const std::string dir);
    
    // Inference
    void Inference_multiDNNs(
        const std::vector<float>& input0, std::vector<double>& output0, int64_t input_count0,
        const std::vector<float>& input1, std::vector<double>& output1, int64_t input_count1,
        const std::vector<float>& input2, std::vector<double>& output2, int64_t input_count2);

};

#endif