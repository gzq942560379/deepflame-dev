#include "matrix_gen_slave_param.h"
#include "assert.h"


void preProcessY_gradY_div_meshV_naive(preProcessY_gradY_div_meshV_param_t* para_p){
    preProcessY_gradY_div_meshV_param_t para;
    CRTS_dma_get(&para, para_p, sizeof(preProcessY_gradY_div_meshV_param_t));

    label nCells = para.nCells;
    scalarPtr gradY_Species = para.gradY_Species;
    constScalarPtr meshVPtr = para.meshVPtr;

    label localCellStart = LOCAL_START(nCells, CRTS_tid, CRTS_MAX_SPE_NUM);
    label localCellEnd = LOCAL_END(nCells, CRTS_tid, CRTS_MAX_SPE_NUM);
    label localCellLen = localCellEnd - localCellStart;
    
    for (label c = localCellStart; c < localCellEnd; ++c) {
        scalar meshVRTmp = 1. / meshVPtr[c];
        gradY_Species[c * 3 + 0] *= meshVRTmp;
        gradY_Species[c * 3 + 1] *= meshVRTmp;
        gradY_Species[c * 3 + 2] *= meshVRTmp;
    }

} 