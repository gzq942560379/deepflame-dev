#include "divMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>
#include "env.H"

namespace Foam{

defineTypeNameAndDebug(divMatrix, 1);

divMatrix::divMatrix(const lduMatrix& ldu):lduMatrix_(ldu),row_block_bit_(row_block_bit), row_block_size_(1 << row_block_bit_){
    diagonal_ = ldu.diagonal();
    symmetric_ = ldu.symmetric();
    asymmetric_ = ldu.asymmetric();

    const auto& lduDiag = ldu.diag();
    const auto& lduLower = ldu.lower();
    const auto& lduUpper = ldu.upper();
    const auto& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu.lduAddr().upperAddr();

    row_ = lduDiag.size();
    block_count_ = DIV_BLOCK_COUNT;
    block_tail_ = DIV_BLOCK_TAIL;
    assert(block_tail_ == 0);
    // fill diag value
    diag_value_.resize(row_);
    for(label i = 0; i < row_; ++i){
        diag_value_[i] = lduDiag[i];
    }
    // compute distance
    // distance range [0 - row + 1, col - 1 - 0] 
    // distance range [1 - row, col - 1] 
    // - 1 + row
    // distance range [0 , row + col - 2]
    // assert (col == row)
    std::vector<label> distance_map(2 * row_ - 1, 0);
    distance_count_ = 0;
    for(label i = 0; i < lduLower.size(); ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        label distance = c - r;
        if(distance_map[distance + row_ - 1] == 0){
            distance_count_ += 1;
            distance_map[distance + row_ - 1] = 1;
        }
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        label distance = c - r;
        if(distance_map[distance + row_ - 1] == 0){
            distance_count_ += 1;
            distance_map[distance + row_ - 1] = 1;
        }
    }
    distance_list_.resize(distance_count_);
    label distance_index = 0;
    for(label i = 0 ; i < distance_map.size(); ++i){
        if(distance_map[i] > 0){
            distance_list_[distance_index] = i - row_ + 1;
            distance_map[i] = distance_index;
            distance_index += 1;
        }else{
            distance_map[i] = -1;
        }
    }
    max_distance_ = 0;
    min_distance_ = 0;
    for(label i = 0; i < distance_list_.size(); ++i){
        max_distance_ = std::max(max_distance_, distance_list_[i]);
        min_distance_ = std::min(min_distance_, distance_list_[i]);
    }
    head_block_count_ = (- min_distance_ + row_block_size_ - 1) / row_block_size_;
    tail_block_count_ = (max_distance_ + row_block_size_ - 1) / row_block_size_;
    // fill off_diag_value
    off_diag_value_.resize(distance_count_ * row_);
    for(label i = 0 ; i < off_diag_value_.size(); ++i){
        off_diag_value_[i] = 0.;
    }
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        label col = lduLowerAddr[i];
        label distance = col - row;
        label divcol = distance_map[distance + row_ - 1];
        assert(divcol != -1);
        assert(divcol < distance_count_);
        label index = DIV_INDEX(row, divcol);
        off_diag_value_[index] = lduLower[i];
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        label col = lduUpperAddr[i];
        label distance = col - row;
        label divcol = distance_map[distance + row_ - 1];
        assert(divcol != -1);
        assert(divcol < distance_count_);
        label index = DIV_INDEX(row, divcol);
        off_diag_value_[index] = lduLower[i];
    }

    Info << "row_ : " << row_ << endl;
    Info << "block_count_ : " << block_count_ << endl;
    Info << "block_tail_ : " << block_tail_ << endl;
    Info << "max_distance_ : " << max_distance_ << endl;
    Info << "min_distance_ : " << min_distance_ << endl;
    Info << "head_block_count_ : " << head_block_count_ << endl;
    Info << "tail_block_count_ : " << tail_block_count_ << endl;
    Info << "distance_list_ : " << distance_list_ << endl;

}

void divMatrix::analyze() const {
}

void divMatrix::check() const {
}


}
