#include "divMatrix_slave_common.h"
#include "divMatrix_slave_param.h"

#define MAX_CELL_LOCAL 8192

void divMatrix_Jacobi_naive(divMatrix_Jacobi_param_t* para){
    const label ZERO = 0;

    scalar* psiPtr = para->psiPtr;
    scalar* psiCopyPtr = para->psiCopyPtr;
    const scalar* bPrimePtr = para->bPrimePtr;
    const scalar* diagPtr = para->diagPtr;
    const scalar* off_diag_value_Ptr = para->off_diag_value_Ptr;
    const label* distance_list_ = para->distance_list_;

    label row_ = para->row_;
    label distance_count_ = para->distance_count_;
    label block_count_ = para->block_count_;
    label row_block_bit_ = para->row_block_bit_;
    label row_block_size_ = para->row_block_size_;

    label block_start = LOCAL_START(block_count_, CRTS_tid, CRTS_MAX_SPE_NUM);
    label block_end = LOCAL_END(block_count_, CRTS_tid, CRTS_MAX_SPE_NUM);

    label row_start =  DIV_BLOCK_START(block_start);
    label row_end =  DIV_BLOCK_START(block_end);
    label row_len = row_end - row_start;
    
    scalar buffer[MAX_CELL_LOCAL];

    for(label brs = 0; brs < row_len; brs += MAX_CELL_LOCAL){
        label bre = slave_min(row_len, brs + MAX_CELL_LOCAL);
        label brl = bre - brs;
        CRTS_dma_get(buffer, psiPtr + row_start + brs, brl * sizeof(double));
        CRTS_dma_put(psiCopyPtr + row_start + brs, buffer, brl * sizeof(double));
    }

    CRTS_ssync_array();

    for(label bi = block_start; bi < block_end; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
        const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            psiPtr_offset[br] = bPrimePtr_offset[br];
        }
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbs >= 0 && distance + rbe <= row_){
                for(label br = 0; br < rbl; ++br){
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = slave_max(col, ZERO);
                    col = slave_min(col, row_ - 1);
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[col];
                }
            }
        }
        for(label br = 0; br < rbl; ++br){
            psiPtr_offset[br] /= diagPtr_offset[br];
        }
    }
}