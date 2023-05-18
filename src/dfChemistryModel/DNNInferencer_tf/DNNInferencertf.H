#ifndef DNNInferencertf_H
#define DNNInferencertf_H

#include <tensorflow/c/c_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string> 
#include <memory>
#include <cstring>

class DNNInferencertf
{
private:
    TF_Graph *graph0_ = TF_NewGraph();
    TF_Graph *graph1_ = TF_NewGraph();
    TF_Graph *graph2_ = TF_NewGraph();
    TF_Session *session0_ = nullptr, *session1_ = nullptr, *session2_ = nullptr;
    TF_Tensor *input_tensor0_[1], *input_tensor1_[1], *input_tensor2_[1],
              *output_tensor0_[1], *output_tensor1_[1], *output_tensor2_[1];
    TF_Output inputs0_[1], inputs1_[1], inputs2_[1], outputs0_[1], outputs1_[1], outputs2_[1];

    TF_Status* status = TF_NewStatus();

public:
    DNNInferencertf();
    DNNInferencertf(const std::vector<char>& input_model_0,const std::vector<char>& input_model_1,const std::vector<char>& input_model_2);
    ~DNNInferencertf();

    // Inference
    std::vector<std::vector<double>> Inference_multiDNNs(const std::vector<std::vector<float>>& DNNinputs, int dimension);
    void Inference0(const std::vector<float>& inputs, std::vector<double>& outputs, int64_t sample_count , int64_t input_dim, int64_t output_dim);
    void Inference1(const std::vector<float>& inputs, std::vector<double>& outputs, int64_t sample_count , int64_t input_dim, int64_t output_dim);
    void Inference2(const std::vector<float>& inputs, std::vector<double>& outputs, int64_t sample_count , int64_t input_dim, int64_t output_dim);

};

#endif