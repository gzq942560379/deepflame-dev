#include "matrix_gen_slave_param.h"
#include "assert.h"

#define MAX_LOCAL_X_LEN 128
#define MAX_LOCAL_Y_LEN 8
#define MAX_LOCAL_Z_LEN 8
#define MAX_FACE_SIZE 3
#define GRADY_DIM 3

#define INDEX2(x, y, ldx) ((x) + (y) * (ldx));
#define INDEX3(x, y, z, ldx, ldy) ((x) + (y) * (ldx) + (z) * (ldx) * (ldy));

#define globalX(lx) ((lx))
#define globalY(ly) ((ly) + (localYStart))
#define globalZ(lz) ((lz) + (localZStart))

#define localIndex(lx, ly, lz) INDEX3(lx, ly, lz, localXLen, localYLen)
#define globalIndex(gx, gy, gz) INDEX3(gx, gy, gz, globalXLen, globalYLen)

#define hasLeftHalo (cid > 0)
#define hasRightHalo (cid < 7)
#define hasUpperHalo (rid > 0)
#define hasDownHalo (rid < 7)

#define THREAD_LEFT (tid - 1)
#define THREAD_RIGHT (tid + 1)
#define THREAD_UPPER (tid - 8)
#define THREAD_DOWN (tid + 8)

// __thread_local scalar local_gradY_Species[2][MAX_LOCAL_X_LEN * MAX_LOCAL_Y_LEN * GRADY_DIM] __attribute__ ((aligned(64)));
// __thread_local scalar local_gradY_Species_Patch_right[MAX_LOCAL_Z_LEN * MAX_LOCAL_X_LEN * GRADY_DIM] __attribute__ ((aligned(64)));
// __thread_local scalar local_gradY_Species_Patch_down[MAX_LOCAL_Y_LEN * MAX_LOCAL_X_LEN * GRADY_DIM] __attribute__ ((aligned(64)));
// __thread_local scalar local_gradY_Species_Patch_left[MAX_LOCAL_Z_LEN * MAX_LOCAL_X_LEN * GRADY_DIM] __attribute__ ((aligned(64)));
// __thread_local scalar local_gradY_Species_Patch_upper[MAX_LOCAL_Y_LEN * MAX_LOCAL_X_LEN * GRADY_DIM] __attribute__ ((aligned(64)));

void preProcessY_gradY_partition_naive(preProcessY_gradY_partition_param_t* para_p){
    preProcessY_gradY_partition_param_t para;
    CRTS_dma_get(&para, para_p, sizeof(preProcessY_gradY_partition_param_t));

    label tid = CRTS_tid;
    label rid = CRTS_rid;
    label cid = CRTS_cid;

    label globalXLen = para.globalXLen;
    label globalYLen = para.globalYLen;
    label globalZLen = para.globalZLen;

    label localXStart = 0;
    label localXLen = para.globalXLen;

    label localYStart = LOCAL_START(globalYLen, cid, 8);
    label localYEnd = LOCAL_END(globalYLen, cid, 8);
    label localYLen = localYEnd - localYStart;

    label localZStart = LOCAL_START(globalZLen, rid, 8);
    label localZEnd = LOCAL_END(globalZLen, rid, 8);
    label localZLen = localZEnd - localZStart;

    constLabelPtr localGlabalFacePtr = para.localGlabalFacePtr[tid];
    constLabelPtr localGlabalFace = para.localGlabalFace[tid];

    constScalarPtr weightsPtr = para.weightsPtr;
    constScalarPtr meshSfPtr = para.meshSfPtr;
    constScalarPtr Yi = para.Yi;

    scalarPtr gradY_Species = para.gradY_Species;

    scalar local_gradY_Species[2][MAX_LOCAL_X_LEN * MAX_LOCAL_Y_LEN * GRADY_DIM];
    scalar local_gradY_Species_Patch_right[MAX_LOCAL_Z_LEN * MAX_LOCAL_X_LEN * GRADY_DIM];
    scalar local_gradY_Species_Patch_down[MAX_LOCAL_Y_LEN * MAX_LOCAL_X_LEN * GRADY_DIM];
    scalar local_gradY_Species_Patch_left[MAX_LOCAL_Z_LEN * MAX_LOCAL_X_LEN * GRADY_DIM];
    scalar local_gradY_Species_Patch_upper[MAX_LOCAL_Y_LEN * MAX_LOCAL_X_LEN * GRADY_DIM];

    label localGlabalFacePtrBuffer[MAX_LOCAL_X_LEN];
    label localGlabalFaceBuffer[MAX_LOCAL_X_LEN * MAX_FACE_SIZE];

    scalar YiBuffer[2][(MAX_LOCAL_Y_LEN + 1) * MAX_LOCAL_X_LEN];

    label YiBufferSize = hasRightHalo ? (localYLen + 1) * localXLen : localYLen * localXLen;

    scalar weightsPtrBuffer[MAX_LOCAL_X_LEN * MAX_FACE_SIZE];
    scalar meshSfPtrBuffer[MAX_LOCAL_X_LEN * GRADY_DIM  * MAX_FACE_SIZE];

    for(label i = 0; i < MAX_LOCAL_X_LEN * MAX_LOCAL_Y_LEN * GRADY_DIM; ++i){
        local_gradY_Species[0][i] = 0;
        local_gradY_Species[1][i] = 0;
        local_gradY_Species_Patch_down[i] = 0;
    }

    for(label i = 0; i < MAX_LOCAL_X_LEN * MAX_LOCAL_Z_LEN * GRADY_DIM; ++i){
        local_gradY_Species_Patch_right[i] = 0;
    }

    label gc_layer_start = globalIndex(0, localYStart, localZStart);
    
    CRTS_dma_get(YiBuffer[0], Yi + gc_layer_start, YiBufferSize * sizeof(scalar));

    for(label lz = 0; lz < localZLen; ++lz){
        label gz = globalZ(lz);
        label cur = lz & 1;
        label next = (lz + 1) & 1;

        scalar* local_gradY_Species_cur = local_gradY_Species[cur];
        scalar* local_gradY_Species_next = local_gradY_Species[next];
        scalar* YiBuffer_cur = YiBuffer[cur];
        scalar* YiBuffer_next = YiBuffer[next];

        // load next layer Yi
        label gc_next_layer_start = globalIndex(0, localYStart, gz + 1);
        if(gz + 1 < globalZLen){
            CRTS_dma_get(YiBuffer_next, (scalar*)Yi + gc_next_layer_start, YiBufferSize * sizeof(scalar));
        }

        for(label ly = 0; ly < localYLen; ++ly){
            label gy = globalY(ly);
            
            label lc_vector_start = localIndex(0, ly, lz);
            CRTS_dma_get(localGlabalFacePtrBuffer, (label*)localGlabalFacePtr + lc_vector_start, (localXLen + 1) * sizeof(label));

            label localGlabalFaceStart = localGlabalFacePtrBuffer[0];
            label localGlabalFaceEnd = localGlabalFacePtrBuffer[localXLen];
            label localGlabalFaceLen = localGlabalFaceEnd - localGlabalFaceStart;
            CRTS_dma_get(localGlabalFaceBuffer, (label*)localGlabalFace + localGlabalFaceStart, localGlabalFaceLen * sizeof(label));

            label globalFaceStart = localGlabalFaceBuffer[0];
            label globalFaceEnd = localGlabalFaceBuffer[localGlabalFaceLen - 1] + 1;
            label globalFaceLen = globalFaceEnd - globalFaceStart;
            CRTS_dma_get(weightsPtrBuffer, (label*)weightsPtr + globalFaceStart, globalFaceLen * sizeof(scalar));

            CRTS_dma_get(meshSfPtrBuffer, (label*)meshSfPtr + globalFaceStart * GRADY_DIM, globalFaceLen * GRADY_DIM * sizeof(scalar));

            for(label lx = 0; lx < localXLen; ++lx){
                label gx = globalX(lx);
                label lc = localIndex(lx, ly, lz);
                label lc_cur = INDEX2(lx, ly, localXLen);
                label gc = globalIndex(gx, gy, gz);
                label gfp = localGlabalFacePtrBuffer[lx] - localGlabalFaceStart;
                scalar ownerYi = Yi[gc];
                if(gx + 1 < globalXLen){
                    label lnc = localIndex(lx + 1, ly, lz);
                    label lnc_cur = INDEX2(lx + 1, ly, localXLen);
                    label gnc = globalIndex(gx + 1, gy, gz);
                    label gf = localGlabalFaceBuffer[gfp];
                    label gf_vector = gf - globalFaceStart;
                    // scalar neighbourYi = Yi[gnc];
                    scalar neighbourYi = YiBuffer_cur[lnc_cur];
                    scalar ssf = (weightsPtrBuffer[gf_vector] * (ownerYi - neighbourYi) + neighbourYi);
                    scalar grad_x = meshSfPtrBuffer[3 * gf_vector + 0] * ssf;
                    scalar grad_y = meshSfPtrBuffer[3 * gf_vector + 1] * ssf;
                    scalar grad_z = meshSfPtrBuffer[3 * gf_vector + 2] * ssf;
                    local_gradY_Species_cur[lc_cur * 3 + 0] += grad_x;
                    local_gradY_Species_cur[lc_cur * 3 + 1] += grad_y;
                    local_gradY_Species_cur[lc_cur * 3 + 2] += grad_z;
                    local_gradY_Species_cur[lnc_cur * 3 + 0] -= grad_x;
                    local_gradY_Species_cur[lnc_cur * 3 + 1] -= grad_y;
                    local_gradY_Species_cur[lnc_cur * 3 + 2] -= grad_z;                
                    gfp += 1;
                }
                
                if(gy + 1 < globalYLen){
                    label gnc = globalIndex(gx, gy + 1, gz);
                    label gf = localGlabalFaceBuffer[gfp];
                    label gf_vector = gf - globalFaceStart;
                    label lnc_cur = INDEX2(lx, ly + 1, localXLen);
                    scalar neighbourYi = YiBuffer_cur[lnc_cur];
                    scalar ssf = (weightsPtrBuffer[gf_vector] * (ownerYi - neighbourYi) + neighbourYi);
                    scalar grad_x = meshSfPtrBuffer[3 * gf_vector + 0] * ssf;
                    scalar grad_y = meshSfPtrBuffer[3 * gf_vector + 1] * ssf;
                    scalar grad_z = meshSfPtrBuffer[3 * gf_vector + 2] * ssf;
                    local_gradY_Species_cur[lc_cur * 3 + 0] += grad_x;
                    local_gradY_Species_cur[lc_cur * 3 + 1] += grad_y;
                    local_gradY_Species_cur[lc_cur * 3 + 2] += grad_z;
                    if(ly + 1 < localYLen){
                        local_gradY_Species_cur[lnc_cur * 3 + 0] -= grad_x;
                        local_gradY_Species_cur[lnc_cur * 3 + 1] -= grad_y;
                        local_gradY_Species_cur[lnc_cur * 3 + 2] -= grad_z;
                    }else{
                        label hi = lx + lz * localXLen;
                        local_gradY_Species_Patch_right[hi * 3 + 0] -= grad_x;
                        local_gradY_Species_Patch_right[hi * 3 + 1] -= grad_y;
                        local_gradY_Species_Patch_right[hi * 3 + 2] -= grad_z;
                    }
                    gfp += 1;
                }

                if(gz + 1 < globalZLen){
                    label gnc = globalIndex(gx, gy, gz + 1);
                    label gf = localGlabalFaceBuffer[gfp];
                    label gf_vector = gf - globalFaceStart;
                    // scalar neighbourYi = Yi[gnc];
                    label lnc_next = INDEX2(lx, ly, localXLen);
                    scalar neighbourYi = YiBuffer_next[lnc_next];
                    scalar ssf = (weightsPtrBuffer[gf_vector] * (ownerYi - neighbourYi) + neighbourYi);
                    scalar grad_x = meshSfPtrBuffer[3 * gf_vector + 0] * ssf;
                    scalar grad_y = meshSfPtrBuffer[3 * gf_vector + 1] * ssf;
                    scalar grad_z = meshSfPtrBuffer[3 * gf_vector + 2] * ssf;
                    local_gradY_Species_cur[lc_cur * 3 + 0] += grad_x;
                    local_gradY_Species_cur[lc_cur * 3 + 1] += grad_y;
                    local_gradY_Species_cur[lc_cur * 3 + 2] += grad_z;
                    if(lz + 1 < localZLen){
                        local_gradY_Species_next[lnc_next * 3 + 0] -= grad_x;
                        local_gradY_Species_next[lnc_next * 3 + 1] -= grad_y;
                        local_gradY_Species_next[lnc_next * 3 + 2] -= grad_z; 
                    }else{
                        label hi = lx + ly * localXLen;
                        local_gradY_Species_Patch_down[hi * 3 + 0] -= grad_x;
                        local_gradY_Species_Patch_down[hi * 3 + 1] -= grad_y;
                        local_gradY_Species_Patch_down[hi * 3 + 2] -= grad_z;
                    }
                    gfp += 1;
                }
            }
        }

        // write gradY_Species
        label gc_start = globalIndex(0, localYStart, gz);
        CRTS_dma_put(gradY_Species + gc_start * 3, local_gradY_Species_cur, localXLen * localYLen * GRADY_DIM * sizeof(scalar));

        // reset curur
        for(label ly = 0; ly < localYLen; ++ly){
            label gy = globalY(ly);
            for(label lx = 0; lx < localXLen; ++lx){
                label gx = globalX(lx);
                label lc_cur = INDEX2(lx, ly, localXLen);
                local_gradY_Species_cur[lc_cur * 3 + 0] = 0;
                local_gradY_Species_cur[lc_cur * 3 + 1] = 0;
                local_gradY_Species_cur[lc_cur * 3 + 2] = 0;
            }
        }
    }

    crts_rply_t rma_rplyl = 0;
    crts_rply_t rma_rplyr = 0;
    unsigned int R_COUNTL = 0;
    unsigned int R_COUNTR = 0;

    CRTS_ssync_row();

    if(hasLeftHalo){
        CRTS_rma_iget(local_gradY_Species_Patch_left, &rma_rplyl, sizeof(scalar) * localXLen * localZLen * GRADY_DIM, THREAD_LEFT, local_gradY_Species_Patch_right, &rma_rplyr);
        R_COUNTL += 1;
        CRTS_rma_wait_value(&rma_rplyl, R_COUNTL);
    }
    if(hasRightHalo){
        R_COUNTR += 1;
        CRTS_rma_wait_value(&rma_rplyr, R_COUNTR);
    }

    CRTS_ssync_col();
    if(hasUpperHalo){
        CRTS_rma_iget(local_gradY_Species_Patch_upper, &rma_rplyl, sizeof(scalar) * localXLen * localYLen * GRADY_DIM, THREAD_UPPER, local_gradY_Species_Patch_down, &rma_rplyr);
        R_COUNTL += 1;
        CRTS_rma_wait_value(&rma_rplyl, R_COUNTL);
    }
    if(hasDownHalo){
        R_COUNTR += 1;
        CRTS_rma_wait_value(&rma_rplyr, R_COUNTR);
    }

    if(hasLeftHalo){
        scalar* local_gradY_Species_Buffer = local_gradY_Species[0];

        label ly = 0;
        label gy = globalY(ly);
        for(label lz = 0; lz < localZLen; ++lz){
            label gz = globalZ(lz);
            label gc_vector_start = globalIndex(0, gy, gz);
            CRTS_dma_get(local_gradY_Species_Buffer, gradY_Species + gc_vector_start * GRADY_DIM, localXLen * GRADY_DIM * sizeof(scalar));
            for(label lx = 0; lx < localXLen; ++lx){
                label gx = globalX(lx);
                label lc = localIndex(lx, ly, lz);
                label gc = globalIndex(gx, gy, gz);
                label hi = lx + lz * localXLen;
                local_gradY_Species_Buffer[lx * 3 + 0] += local_gradY_Species_Patch_left[hi * 3 + 0];
                local_gradY_Species_Buffer[lx * 3 + 1] += local_gradY_Species_Patch_left[hi * 3 + 1];
                local_gradY_Species_Buffer[lx * 3 + 2] += local_gradY_Species_Patch_left[hi * 3 + 2];
            }
            CRTS_dma_put(gradY_Species + gc_vector_start * GRADY_DIM, local_gradY_Species_Buffer, localXLen * GRADY_DIM * sizeof(scalar));
        }
    }
    if(hasUpperHalo){
        scalar* local_gradY_Species_Buffer = local_gradY_Species[0];

        label lz = 0;
        label gz = globalZ(lz);
        label gc_vector_start = globalIndex(0, localYStart, gz);
        CRTS_dma_get(local_gradY_Species_Buffer, gradY_Species + gc_vector_start * GRADY_DIM, localYLen * localXLen * GRADY_DIM * sizeof(scalar));
        for(label ly = 0; ly < localYLen; ++ly){
            label gy = globalY(ly);
            for(label lx = 0; lx < localXLen; ++lx){
                label gx = globalX(lx);
                label lc = localIndex(lx, ly, lz);
                label gc = globalIndex(gx, gy, gz);
                label hi = lx + ly * localXLen;
                local_gradY_Species_Buffer[hi * 3 + 0] += local_gradY_Species_Patch_upper[hi * 3 + 0];
                local_gradY_Species_Buffer[hi * 3 + 1] += local_gradY_Species_Patch_upper[hi * 3 + 1];
                local_gradY_Species_Buffer[hi * 3 + 2] += local_gradY_Species_Patch_upper[hi * 3 + 2];
            }
        }
        CRTS_dma_put(gradY_Species + gc_vector_start * GRADY_DIM, local_gradY_Species_Buffer, localYLen * localXLen * GRADY_DIM * sizeof(scalar));
    }
} 