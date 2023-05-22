#include "DNNInferencer_blas.H"
#include <cassert>

DNNInferencer_blas::DNNInferencer_blas() {}

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

void DNNInferencer_blas::alloc_buffer(int64_t total_samples){
    if(buffer_alloced_){
        if(total_samples <= total_samples_){
            return;
        }else{
            assert(false);
        }
    }
    total_samples_ = total_samples;
    buffer_alloced_ = true;
    output_buffer_.emplace_back(std::vector<float>(total_samples * layers_[1]));
    output_buffer_.emplace_back(std::vector<float>(total_samples * layers_[2]));
    output_buffer_.emplace_back(std::vector<float>(total_samples * layers_[3]));
    output_buffer_.emplace_back(std::vector<float>(total_samples * layers_[4]));

}

void DNNInferencer_blas::Inference_multiDNNs(
    const std::vector<float>& input0, std::vector<double>& output0, int64_t input_count0,
    const std::vector<float>& input1, std::vector<double>& output1, int64_t input_count1,
    const std::vector<float>& input2, std::vector<double>& output2, int64_t input_count2){
    
    assert(input_count0 + input_count1 + input_count2 <= total_samples_);

    assert(buffer_alloced_);

    if(input_count0 > 0){
        int offset0 = 0;
        Tensor<float> input0_0({input_count0, layers_[0]}, const_cast<float*>(input0.data()));
        Tensor<float> output0_1({input_count0, layers_[1]}, output_buffer_[0].data() + offset0 * layers_[1]);
        Tensor<float> output0_2({input_count0, layers_[2]}, output_buffer_[1].data() + offset0 * layers_[2]);
        Tensor<float> output0_3({input_count0, layers_[3]}, output_buffer_[2].data() + offset0 * layers_[3]);
        Tensor<float> output0_4({input_count0, layers_[4]}, output_buffer_[3].data() + offset0 * layers_[4]);


        model0_[0]->forward(input0_0, output0_1);
        model0_[1]->forward(output0_1, output0_2);
        model0_[2]->forward(output0_2, output0_3);
        model0_[3]->forward(output0_3, output0_4);

        assert(output0_4.element_num() == output0.size());
        output0.resize(output0_4.element_num());
        for(int i = 0; i < output0_4.element_num(); ++i){
            output0[i] = output0_4.data()[i];
        }

    }

    if(input_count1 > 0){
        int offset1 = input_count0;
        Tensor<float> input1_0({input_count1, layers_[0]}, const_cast<float*>(input1.data()));
        Tensor<float> output1_1({input_count1, layers_[1]}, output_buffer_[0].data() + offset1 * layers_[1]);
        Tensor<float> output1_2({input_count1, layers_[2]}, output_buffer_[1].data() + offset1 * layers_[2]);
        Tensor<float> output1_3({input_count1, layers_[3]}, output_buffer_[2].data() + offset1 * layers_[3]);
        Tensor<float> output1_4({input_count1, layers_[4]}, output_buffer_[3].data() + offset1 * layers_[4]);

        model1_[0]->forward(input1_0, output1_1);
        model1_[1]->forward(output1_1, output1_2);
        model1_[2]->forward(output1_2, output1_3);
        model1_[3]->forward(output1_3, output1_4);

        assert(output1_4.element_num() == output1.size());
        output1.resize(output1_4.element_num());
        for(int i = 0; i < output1_4.element_num(); ++i){
            output1[i] = output1_4.data()[i];
        }
    }

    if(input_count2 > 0){
        int offset2 = input_count0 + input_count1;
        Tensor<float> input2_0({input_count2, layers_[0]}, const_cast<float*>(input2.data()));
        Tensor<float> output2_1({input_count2, layers_[1]}, output_buffer_[0].data() + offset2 * layers_[1]);
        Tensor<float> output2_2({input_count2, layers_[2]}, output_buffer_[1].data() + offset2 * layers_[2]);
        Tensor<float> output2_3({input_count2, layers_[3]}, output_buffer_[2].data() + offset2 * layers_[3]);
        Tensor<float> output2_4({input_count2, layers_[4]}, output_buffer_[3].data() + offset2 * layers_[4]);

        model2_[0]->forward(input2_0, output2_1);
        model2_[1]->forward(output2_1, output2_2);
        model2_[2]->forward(output2_2, output2_3);
        model2_[3]->forward(output2_3, output2_4);

        assert(output2_4.element_num() == output2.size());
        output2.resize(output2_4.element_num());
        for(int i = 0; i < output2_4.element_num(); ++i){
            output2[i] = output2_4.data()[i];
        }
    }

}



