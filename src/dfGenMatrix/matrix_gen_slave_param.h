#pragma once
#include <crts.h>
#include "slave_utils.h"



#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
    // label localXLen[CRTS_MAX_SPE_NUM];
    // label localYLen[CRTS_MAX_SPE_NUM];
    // label localZLen[CRTS_MAX_SPE_NUM];
    // label localXStart[CRTS_MAX_SPE_NUM];
    // label localYStart[CRTS_MAX_SPE_NUM];
    // label localZStart[CRTS_MAX_SPE_NUM];
    constLabelPtr localGlabalFacePtr[CRTS_MAX_SPE_NUM];
    constLabelPtr localGlabalFace[CRTS_MAX_SPE_NUM];
    label globalXLen;
    label globalYLen;
    label globalZLen;
    constScalarPtr weightsPtr;
    constScalarPtr meshSfPtr;
    constScalarPtr Yi;
    scalarPtr gradY_Species; // output
} preProcessY_gradY_partition_param_t;

typedef struct{
    label nCells;
    scalarPtr gradY_Species;
    constScalarPtr meshVPtr;
} preProcessY_gradY_div_meshV_param_t;

#ifdef __cplusplus
}
#endif
