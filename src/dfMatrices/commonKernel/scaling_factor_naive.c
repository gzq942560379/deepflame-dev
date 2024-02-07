#include "common_slave_function.h"
#include "common_slave_param.h"

#define MAX_CELL_LOCAL 1024

void scaling_factor_naive(scaling_factor_param_t* para_p){
    scaling_factor_param_t para;

    CRTS_dma_get(&para, para_p, sizeof(scaling_factor_param_t));

    scalar* scalingFactorNum_p = para.scalingFactorNum_p;
    scalar* scalingFactorDenom_p = para.scalingFactorDenom_p;
    const scalar* sourcePtr = para.sourcePtr;
    const scalar* AcfPtr = para.AcfPtr;
    const scalar* fieldPtr = para.fieldPtr;
    const label nCells = para.nCells;

    label local_start = LOCAL_START(nCells, CRTS_tid, CRTS_MAX_SPE_NUM);
    label local_end = LOCAL_END(nCells, CRTS_tid, CRTS_MAX_SPE_NUM);
    label local_len = local_end - local_start;

    const scalar* local_sourcePtr = sourcePtr + local_start;
    const scalar* local_AcfPtr = AcfPtr + local_start;
    const scalar* local_fieldPtr = fieldPtr + local_start;

    scalar sourcePtr_buffer[MAX_CELL_LOCAL];
    scalar AcfPtr_buffer[MAX_CELL_LOCAL];
    scalar fieldPtr_buffer[MAX_CELL_LOCAL];

    label block_count = slave_div_ceil(local_len, MAX_CELL_LOCAL);

    scalar scalingFactorNum = 0.;
    scalar scalingFactorDenom = 0.;

    for(label bi = 0; bi < block_count; ++bi){
        label brs = bi * MAX_CELL_LOCAL;
        label bre = slave_min(local_len, brs + MAX_CELL_LOCAL);
        label brl = bre - brs;
        CRTS_dma_get(sourcePtr_buffer, (scalar*)local_sourcePtr + brs, brl * sizeof(double));
        CRTS_dma_get(AcfPtr_buffer, (scalar*)local_AcfPtr + brs, brl * sizeof(double));
        CRTS_dma_get(fieldPtr_buffer, (scalar*)local_fieldPtr + brs, brl * sizeof(double));
        for(label i = 0; i < brl; ++i){
            scalingFactorNum += sourcePtr_buffer[i] * fieldPtr_buffer[i];
            scalingFactorDenom += AcfPtr_buffer[i] * fieldPtr_buffer[i];
        }
    }

    scalingFactorNum = reduce_rma(scalingFactorNum);
    scalingFactorDenom = reduce_rma(scalingFactorDenom);

    if(CRTS_tid == 0){
        CRTS_dma_put(scalingFactorNum_p, &scalingFactorNum, sizeof(scalar));
        CRTS_dma_put(scalingFactorDenom_p, &scalingFactorDenom, sizeof(scalar));
    }
}