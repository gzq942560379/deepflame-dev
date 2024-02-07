#include "common_slave_param.h"

#define MAX_CELL_LOCAL 1024

void scaling_update_naive(scaling_update_param_t* para_p){
    scaling_update_param_t para;

    CRTS_dma_get(&para, para_p, sizeof(scaling_update_param_t));

    scalar* fieldPtr = para.fieldPtr;
    const scalar sf = para.sf;
    const scalar* sourcePtr = para.sourcePtr;
    const scalar* AcfPtr = para.AcfPtr;
    const scalar* diagPtr = para.diagPtr;
    label nCells = para.nCells;

    label local_start = LOCAL_START(nCells, CRTS_tid, CRTS_MAX_SPE_NUM);
    label local_end = LOCAL_END(nCells, CRTS_tid, CRTS_MAX_SPE_NUM);
    label local_len = local_end - local_start;

    scalar* local_fieldPtr = fieldPtr + local_start;
    const scalar* local_sourcePtr = sourcePtr + local_start;
    const scalar* local_AcfPtr = AcfPtr + local_start;
    const scalar* local_diagPtr = diagPtr + local_start;

    scalar fieldPtr_buffer[MAX_CELL_LOCAL];
    scalar sourcePtr_buffer[MAX_CELL_LOCAL];
    scalar AcfPtr_buffer[MAX_CELL_LOCAL];
    scalar diagPtr_buffer[MAX_CELL_LOCAL];

    label block_count = slave_div_ceil(local_len, MAX_CELL_LOCAL);
    
    for(label bi = 0; bi < block_count; ++bi){
        label brs = bi * MAX_CELL_LOCAL;
        label bre = slave_min(local_len, brs + MAX_CELL_LOCAL);
        label brl = bre - brs;
        CRTS_dma_get(fieldPtr_buffer, (scalar*)local_fieldPtr + brs, brl * sizeof(double));
        CRTS_dma_get(sourcePtr_buffer, (scalar*)local_sourcePtr + brs, brl * sizeof(double));
        CRTS_dma_get(AcfPtr_buffer, (scalar*)local_AcfPtr + brs, brl * sizeof(double));
        CRTS_dma_get(diagPtr_buffer, (scalar*)local_diagPtr + brs, brl * sizeof(double));
        for(label i = 0; i < brl; ++i){
            fieldPtr_buffer[i] = sf * fieldPtr_buffer[i] + (sourcePtr_buffer[i] - sf * AcfPtr_buffer[i]) / diagPtr_buffer[i];
        }
        CRTS_dma_put(local_fieldPtr + brs, fieldPtr_buffer, brl * sizeof(double));
    }
}