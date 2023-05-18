#include "DNNInferencertf.H"
#include <cassert>

DNNInferencertf::DNNInferencertf() {}

#define CHEAK_TF(status){\
    if(TF_GetCode(status) != TF_OK){\
        std::cerr << "Tensorflow Error : " << TF_Message(status) << "at line " << __LINE__ << std::endl << std::flush;\
        std::abort();\
    }\
} 

DNNInferencertf::DNNInferencertf(const std::vector<char>& input_model_0,const std::vector<char>& input_model_1,const std::vector<char>& input_model_2)
{
    TF_Buffer *buffer0_ = TF_NewBuffer();
    TF_Buffer *buffer1_ = TF_NewBuffer();
    TF_Buffer *buffer2_ = TF_NewBuffer();
    TF_ImportGraphDefOptions* Graph_options0 = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptions* Graph_options1 = TF_NewImportGraphDefOptions();
    TF_ImportGraphDefOptions* Graph_options2 = TF_NewImportGraphDefOptions();

    // Create a TF_Buffer object containing the model data
    buffer0_->data = input_model_0.data();
    buffer0_->length = input_model_0.size();

    buffer1_->data = input_model_1.data();
    buffer1_->length = input_model_1.size();

    buffer2_->data = input_model_2.data();
    buffer2_->length = input_model_2.size();

    // Load the model from the buffer
    TF_GraphImportGraphDef(graph0_, buffer0_, Graph_options0, status);
    CHEAK_TF(status);
    TF_DeleteImportGraphDefOptions(Graph_options0);
    TF_DeleteBuffer(buffer0_);

    TF_GraphImportGraphDef(graph1_, buffer1_, Graph_options1, status);
    CHEAK_TF(status);
    TF_DeleteImportGraphDefOptions(Graph_options1);
    TF_DeleteBuffer(buffer1_);

    TF_GraphImportGraphDef(graph2_, buffer2_, Graph_options2, status);
    CHEAK_TF(status);
    TF_DeleteImportGraphDefOptions(Graph_options2);
    TF_DeleteBuffer(buffer2_);


    TF_SessionOptions* session_options0 = TF_NewSessionOptions();
    TF_SessionOptions* session_options1 = TF_NewSessionOptions();
    TF_SessionOptions* session_options2 = TF_NewSessionOptions();
    // create session
    session0_ = TF_NewSession(graph0_, session_options0, status);
    CHEAK_TF(status);
    session1_ = TF_NewSession(graph1_, session_options1, status);
    CHEAK_TF(status);
    session2_ = TF_NewSession(graph2_, session_options2, status);
    CHEAK_TF(status);

    inputs0_[0] = {TF_GraphOperationByName(graph0_, "input"), 0};
    inputs1_[0] = {TF_GraphOperationByName(graph1_, "input"), 0};
    inputs2_[0] = {TF_GraphOperationByName(graph2_, "input"), 0};
    outputs0_[0] = {TF_GraphOperationByName(graph0_, "add_9"), 0};
    outputs1_[0] = {TF_GraphOperationByName(graph1_, "add_9"), 0};
    outputs2_[0] = {TF_GraphOperationByName(graph2_, "add_9"), 0};

    TF_DeleteSessionOptions(session_options0);
    TF_DeleteSessionOptions(session_options1);
    TF_DeleteSessionOptions(session_options2);

}

DNNInferencertf::~DNNInferencertf(){
    TF_DeleteGraph(graph0_);
    TF_DeleteGraph(graph1_);
    TF_DeleteGraph(graph2_);
    TF_CloseSession(session0_,status);
    TF_CloseSession(session1_,status);
    TF_CloseSession(session2_,status);
    TF_DeleteSession(session0_,status);
    TF_DeleteSession(session1_,status);
    TF_DeleteSession(session2_,status);
    TF_DeleteStatus(status);
}

// std::vector<std::vector<double>> DNNInferencertf::Inference_multiDNNs(const std::vector<std::vector<float>>& DNNinputs, int dimension)
// {
//     std::cout << "Inference_multiDNNs 1 " << std::endl;
//     std::vector<int64_t> input_shape_NN0_ = {{static_cast<int64_t>(DNNinputs[0].size()/dimension), dimension}};
//     std::cout << "Inference_multiDNNs 1.1 " << std::endl;
//     // construct input tensor
//     input_tensor0_ = TF_AllocateTensor(TF_FLOAT, input_shape_NN0_.data(), input_shape_NN0_.size(), DNNinputs[0].size() * sizeof(float));
//     std::cout << "Inference_multiDNNs 1.2 " << std::endl;
//     std::memcpy(TF_TensorData(input_tensor0_), DNNinputs[0].data(), DNNinputs[0].size() * sizeof(float));
//     std::cout << "Inference_multiDNNs 1.3 " << std::endl;
//     // run inference
//     inputs0_ = {TF_GraphOperationByName(graph0_, "input"), 0};
//     std::cout << "Inference_multiDNNs 1.4 " << std::endl;
//     outputs0_ = {TF_GraphOperationByName(graph0_, "add_9"), 0};
//     std::cout << "Inference_multiDNNs 1.5 " << std::endl;
//     TF_SessionRun(session0_, nullptr, &inputs0_, &input_tensor0_, 1, &outputs0_, &output_tensor0_, 1, nullptr, 0, nullptr, status);
//     CHEAK_TF(status);
//     std::cout << "Inference_multiDNNs 1.6 " << std::endl;
//     const auto result0_ = static_cast<float*>(TF_TensorData(output_tensor0_));
//     std::cout << "Inference_multiDNNs 1.7 " << std::endl;
//     std::vector<double> result0_vec_(result0_, result0_ + DNNinputs[0].size());

//     std::cout << "Inference_multiDNNs 2 " << std::endl;

//     std::vector<int64_t> input_shape_NN1_ = {{static_cast<int64_t>(DNNinputs[1].size()/dimension), dimension}};
//     std::cout << "Inference_multiDNNs 2.1 " << std::endl;
//     input_tensor1_ = TF_AllocateTensor(TF_FLOAT, input_shape_NN1_.data(), input_shape_NN1_.size(), DNNinputs[1].size() * sizeof(float));
//     std::cout << "Inference_multiDNNs 2.2 " << std::endl;
//     std::memcpy(TF_TensorData(input_tensor1_), DNNinputs[1].data(), DNNinputs[1].size() * sizeof(float));
//     std::cout << "Inference_multiDNNs 2.3 " << std::endl;
//     inputs1_ = {TF_GraphOperationByName(graph1_, "input"), 0};
//     std::cout << "Inference_multiDNNs 2.4 " << std::endl;
//     outputs1_ = {TF_GraphOperationByName(graph1_, "add_9"), 0};
//     std::cout << "Inference_multiDNNs 2.5 " << std::endl;
//     TF_SessionRun(session1_, nullptr, &inputs1_, &input_tensor1_, 1, &outputs1_, &output_tensor1_, 1, nullptr, 0, nullptr, status);
//     CHEAK_TF(status);
//     std::cout << "Inference_multiDNNs 2.6 " << std::endl;
//     const auto result1_ = static_cast<float*>(TF_TensorData(output_tensor1_));
//     std::cout << "Inference_multiDNNs 2.7 " << std::endl;
//     std::vector<double> result1_vec_(result1_, result1_ + DNNinputs[1].size());

//     std::cout << "Inference_multiDNNs 3 " << std::endl;

//     std::vector<int64_t> input_shape_NN2_ = {{static_cast<int64_t>(DNNinputs[2].size()/dimension), dimension}};
//     std::cout << "Inference_multiDNNs 3.1 " << std::endl;
//     input_tensor2_ = TF_AllocateTensor(TF_FLOAT, input_shape_NN2_.data(), input_shape_NN2_.size(), DNNinputs[2].size() * sizeof(float));
//     std::cout << "Inference_multiDNNs 3.2 " << std::endl;
//     std::memcpy(TF_TensorData(input_tensor2_), DNNinputs[2].data(), DNNinputs[2].size() * sizeof(float));
//     std::cout << "Inference_multiDNNs 3.3 " << std::endl;
//     inputs2_ = {TF_GraphOperationByName(graph2_, "input"), 0};
//     std::cout << "Inference_multiDNNs 3.4 " << std::endl;
//     outputs2_ = {TF_GraphOperationByName(graph2_, "add_9"), 0};
//     std::cout << "Inference_multiDNNs 3.5 " << std::endl;
//     TF_SessionRun(session2_, nullptr, &inputs2_, &input_tensor2_, 1, &outputs2_, &output_tensor2_, 1, nullptr, 0, nullptr, status);
//     CHEAK_TF(status);
//     std::cout << "Inference_multiDNNs 3.6 " << std::endl;
//     const auto result2_ = static_cast<float*>(TF_TensorData(output_tensor2_));
//     std::cout << "Inference_multiDNNs 3.7 " << std::endl;
//     std::vector<double> result2_vec_(result2_, result2_ + DNNinputs[2].size());

//     std::cout << "Inference_multiDNNs 4 " << std::endl;

//     std::vector<std::vector<double>> results = {result0_vec_, result1_vec_, result2_vec_};
//     return results;
// }

void DNNInferencertf::Inference0(const std::vector<float>& inputs, std::vector<double>& outputs, int64_t sample_count , int64_t input_dim, int64_t output_dim)
{
    if(inputs.empty())  return;

    assert(inputs.size() == sample_count * input_dim);
    std::vector<int64_t> input_shape_NN0_ = {sample_count, input_dim};
    std::vector<int64_t> output_shape_NN0_ = {sample_count, output_dim};
    // construct input tensor
    input_tensor0_[0] = TF_AllocateTensor(TF_FLOAT, input_shape_NN0_.data(), input_shape_NN0_.size(), sample_count * input_dim * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor0_[0]), inputs.data(), inputs.size() * sizeof(float));
    // run inference

    TF_SessionRun(session0_, nullptr, inputs0_, input_tensor0_, 1, outputs0_, output_tensor0_, 1, nullptr, 0, nullptr, status);
    CHEAK_TF(status);
    const auto result0_ = static_cast<float*>(TF_TensorData(output_tensor0_[0]));
    size_t output_size = sample_count * output_dim;
    outputs.resize(output_size);
    for(int i = 0; i < output_size ; ++i){
        double tmp = result0_[i];
        outputs[i] = tmp;
    }
    TF_DeleteTensor(input_tensor0_[0]);
    TF_DeleteTensor(output_tensor0_[0]);

}

void DNNInferencertf::Inference1(const std::vector<float>& inputs, std::vector<double>& outputs, int64_t sample_count , int64_t input_dim, int64_t output_dim)
{
    if(inputs.empty())  return;
    assert(inputs.size() == sample_count * input_dim);

    std::vector<int64_t> input_shape_NN1_ = {sample_count, input_dim};
    // construct input tensor
    input_tensor1_[0] = TF_AllocateTensor(TF_FLOAT, input_shape_NN1_.data(), input_shape_NN1_.size(), inputs.size() * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor1_[0]), inputs.data(), inputs.size() * sizeof(float));
    // run inference
    TF_SessionRun(session1_, nullptr, inputs1_, input_tensor1_, 1, outputs1_, output_tensor1_, 1, nullptr, 0, nullptr, status);
    CHEAK_TF(status);
    const auto result1_ = static_cast<float*>(TF_TensorData(output_tensor1_[0]));
    size_t output_size = sample_count * output_dim;
    outputs.resize(output_size);
    for(int i = 0; i < output_size ; ++i){
        outputs[i] = result1_[i];
    }
    TF_DeleteTensor(input_tensor1_[0]);
    TF_DeleteTensor(output_tensor1_[0]);
    
}

void DNNInferencertf::Inference2(const std::vector<float>& inputs, std::vector<double>& outputs, int64_t sample_count , int64_t input_dim, int64_t output_dim)
{
    if(inputs.empty())  return;
    assert(inputs.size() == sample_count * input_dim);

    std::vector<int64_t> input_shape_NN2_ = {sample_count, input_dim};
    // construct input tensor
    input_tensor2_[0] = TF_AllocateTensor(TF_FLOAT, input_shape_NN2_.data(), input_shape_NN2_.size(), inputs.size() * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor2_[0]), inputs.data(), inputs.size() * sizeof(float));
    // run inference
    TF_SessionRun(session2_, nullptr, inputs2_, input_tensor2_, 1, outputs2_, output_tensor2_, 1, nullptr, 0, nullptr, status);
    CHEAK_TF(status);
    const auto result2_ = static_cast<float*>(TF_TensorData(output_tensor2_[0]));
    size_t output_size = sample_count * output_dim;
    outputs.resize(output_size);
    for(int i = 0; i < output_size ; ++i){
        outputs[i] = result2_[i];
    }
    TF_DeleteTensor(input_tensor2_[0]);
    TF_DeleteTensor(output_tensor2_[0]);
}