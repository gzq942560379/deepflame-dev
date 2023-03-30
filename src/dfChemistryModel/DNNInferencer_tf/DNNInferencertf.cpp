#include "DNNInferencertf.H"

DNNInferencertf::DNNInferencertf() {}

DNNInferencertf::DNNInferencertf(std::vector<char> input_model_0, std::vector<char> input_model_1, std::vector<char> input_model_2)
{
    // Create a TF_Buffer object containing the model data
    buffer0_->data = input_model_0.data();
    buffer0_->length = input_model_0.size();

    buffer1_->data = input_model_1.data();
    buffer1_->length = input_model_1.size();

    buffer2_->data = input_model_2.data();
    buffer2_->length = input_model_2.size();

    // Load the model from the buffer
    TF_GraphImportGraphDef(graph0_, buffer0_, Graph_options0, status);
    TF_DeleteImportGraphDefOptions(Graph_options0);
    TF_DeleteBuffer(buffer0_);

    TF_GraphImportGraphDef(graph1_, buffer1_, Graph_options1, status);
    TF_DeleteImportGraphDefOptions(Graph_options1);
    TF_DeleteBuffer(buffer1_);

    TF_GraphImportGraphDef(graph2_, buffer2_, Graph_options2, status);
    TF_DeleteImportGraphDefOptions(Graph_options2);
    TF_DeleteBuffer(buffer2_);

    // create session
    session0_ = TF_NewSession(graph0_, session_options0, status);
    session1_ = TF_NewSession(graph1_, session_options1, status);
    session2_ = TF_NewSession(graph2_, session_options2, status);
}

DNNInferencertf::~DNNInferencertf() {}


std::vector<std::vector<double>> DNNInferencertf::Inference_multiDNNs(std::vector<std::vector<float>> DNNinputs, int dimension)
{
    std::vector<int64_t> input_shape_NN0_ = {{DNNinputs[0].size()/dimension, dimension}};
    std::vector<int64_t> input_shape_NN1_ = {{DNNinputs[1].size()/dimension, dimension}};
    std::vector<int64_t> input_shape_NN2_ = {{DNNinputs[2].size()/dimension, dimension}};

    // construct input tensor
    input_tensor0_ = TF_AllocateTensor(TF_FLOAT, input_shape_NN0_.data(), input_shape_NN0_.size(), DNNinputs[0].size() * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor0_), DNNinputs[0].data(), DNNinputs[0].size() * sizeof(float));
    input_tensor1_ = TF_AllocateTensor(TF_FLOAT, input_shape_NN1_.data(), input_shape_NN1_.size(), DNNinputs[1].size() * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor1_), DNNinputs[1].data(), DNNinputs[1].size() * sizeof(float));
    input_tensor2_ = TF_AllocateTensor(TF_FLOAT, input_shape_NN2_.data(), input_shape_NN2_.size(), DNNinputs[2].size() * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor2_), DNNinputs[2].data(), DNNinputs[2].size() * sizeof(float));

    // run inference
    inputs0_ = {TF_GraphOperationByName(graph0_, "input"), 0};
    outputs0_ = {TF_GraphOperationByName(graph0_, "add_9"), 0};

    inputs1_ = {TF_GraphOperationByName(graph1_, "input"), 0};
    outputs1_ = {TF_GraphOperationByName(graph1_, "add_9"), 0};

    inputs2_ = {TF_GraphOperationByName(graph2_, "input"), 0};
    outputs2_ = {TF_GraphOperationByName(graph2_, "add_9"), 0};

    TF_SessionRun(session0_, nullptr, &inputs0_, &input_tensor0_, 1, &outputs0_, &output_tensor0_, 1, nullptr, 0, nullptr, status);
    TF_SessionRun(session1_, nullptr, &inputs1_, &input_tensor1_, 1, &outputs1_, &output_tensor1_, 1, nullptr, 0, nullptr, status);
    TF_SessionRun(session2_, nullptr, &inputs2_, &input_tensor2_, 1, &outputs2_, &output_tensor2_, 1, nullptr, 0, nullptr, status);

    const auto result0_ = static_cast<float*>(TF_TensorData(output_tensor0_));
    const auto result1_ = static_cast<float*>(TF_TensorData(output_tensor1_));
    const auto result2_ = static_cast<float*>(TF_TensorData(output_tensor2_));

    std::vector<double> result0_vec_(result0_, result0_ + DNNinputs[0].size());
    std::vector<double> result1_vec_(result1_, result1_ + DNNinputs[1].size());
    std::vector<double> result2_vec_(result2_, result2_ + DNNinputs[2].size());

    std::vector<std::vector<double>> results = {result0_vec_, result1_vec_, result2_vec_};

    return results;
}
