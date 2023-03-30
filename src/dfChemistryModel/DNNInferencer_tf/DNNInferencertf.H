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
    TF_Tensor *input_tensor0_, *input_tensor1_, *input_tensor2_,
              *output_tensor0_=nullptr, *output_tensor1_=nullptr, *output_tensor2_=nullptr;
    TF_Output inputs0_, inputs1_, inputs2_, outputs0_, outputs1_, outputs2_;
    TF_Buffer *buffer0_ = TF_NewBuffer();
    TF_Buffer *buffer1_ = TF_NewBuffer();
    TF_Buffer *buffer2_ = TF_NewBuffer();

    TF_ImportGraphDefOptions* Graph_options0 = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptions* Graph_options1 = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptions* Graph_options2 = TF_NewImportGraphDefOptions();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* session_options0 = TF_NewSessionOptions();
    TF_SessionOptions* session_options1 = TF_NewSessionOptions();
    TF_SessionOptions* session_options2 = TF_NewSessionOptions();
    double time_preInf = 0;
    double time_Inference = 0;
    double time_postInf = 0;
    double time_hot = 0;

public:
    DNNInferencertf();
    DNNInferencertf(std::vector<char> input_model_0, std::vector<char> input_model_1, std::vector<char> input_model_2);
    ~DNNInferencertf();

    // Inference
    std::vector<std::vector<double>> Inference_multiDNNs(std::vector<std::vector<float>> DNNinputs, int dimension);
};

#endif