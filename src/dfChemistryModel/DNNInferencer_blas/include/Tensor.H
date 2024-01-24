#pragma once
#include <cstdint>
#include <cstdbool>
#include <numeric>
#include <string>
#include <vector>

void* blasdnn_alloc(size_t size){
#ifdef __sw_64__
    return libc_aligned_malloc(size);
#else
    return aligned_alloc(64, size);
#endif
}

void blasdnn_free(void* ptr){
#ifdef __sw_64__
    return libc_aligned_free(ptr);
#else
    return free(ptr);
#endif
}

template<typename DataType>
class Tensor{
private:
    std::vector<int64_t> shape_;
    int64_t element_num_;
    DataType* data_;
    bool owner_;
public:
    Tensor(std::vector<int64_t> shape):
        shape_(shape),
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_((DataType*)blasdnn_alloc(element_num_ * sizeof(DataType))),
        owner_(true){};

    Tensor(std::vector<int64_t> shape, DataType* data):
        shape_(shape), 
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_(data), 
        owner_(false){};
    ~Tensor(){
        if(owner_)  blasdnn_free((void*)data_);
    }
    int64_t dim_num() const {return shape_.size();};
    int64_t element_num() const {return element_num_;};
    int64_t bytes_num() const {return element_num_ * sizeof(DataType);};
    int64_t dim(int64_t i) const {return shape_[i];};
    const DataType* data() const {return data_;};
    DataType* data() {return data_;};
};

template class Tensor<float>;
template class Tensor<double>;