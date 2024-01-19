#include "divMatrix_slave_common.h"
#include "divMatrix_slave_param.h"

void divMatrix_SpMV_naive(divMatrix_SpMV_param_t* para){
    scalar* ApsiPtr = para->ApsiPtr;
    scalar* psiPtr = para->psiPtr;
    scalar* diagPtr = para->diagPtr;
    scalar* off_diag_value_Ptr = para->off_diag_value_Ptr;
    label* distance_list_ = para->distance_list_;

    label row_ = para->row_;
    label distance_count_ = para->distance_count_;
    label block_count_ = para->block_count_;
    label row_block_bit_ = para->row_block_bit_;
    label row_block_size_ = para->row_block_size_;
    

    label block_start = CRTS_tid * block_count_ / CRTS_MAX_SPE_NUM;
    label block_end = (CRTS_tid + 1) * block_count_ / CRTS_MAX_SPE_NUM;

    for(label bi = block_start; bi < block_end; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            ApsiPtr_offset[br] = diagPtr_offset[br] * psiPtr[rbs + br];
        }
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbs >= 0 && distance + rbe <= row_){
                for(label br = 0; br < rbl; ++br){
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = slave_max(col, (label)0);
                    col = slave_min(col, row_ - 1);
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[col];
                }       
            }
        }
    }

}