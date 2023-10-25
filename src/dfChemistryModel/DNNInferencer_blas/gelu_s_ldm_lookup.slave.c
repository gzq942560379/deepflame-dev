#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"
#include "gelu_s_slave_table.h"

#define GELU_S_BLOCK_SIZE (8192)
#define min(x,y) ((x) < (y) ? (x) : (y))

void gelu_s_ldm_lookup(gelu_s_param_t *para_p){
    
    int64_t len = para_p->len;
    float* data = para_p->data;

    int64_t start = CRTS_tid * len / CRTS_MAX_SPE_NUM;
    int64_t end = (CRTS_tid + 1) * len / CRTS_MAX_SPE_NUM;

    float data_ldm[GELU_S_BLOCK_SIZE] __attribute__ ((aligned(64)));

    for (int64_t bs = start; bs < end; bs += GELU_S_BLOCK_SIZE){
        int64_t be = min(end, bs + GELU_S_BLOCK_SIZE);
        int64_t bl = be - bs;
        CRTS_dma_get(data_ldm, data + bs, bl * sizeof(float));
        for(int64_t i = 0; i < bl; ++i){
            float x = data_ldm[i];
            if(x < -5.){
                data_ldm[i] = 0.f;
            }else if(x > 5.){
                data_ldm[i] = x;
            }else{
                // data_ldm[i] = 0.5 * x * (1.f + tanhf(sqrtf(2.f / M_PI) * (x + 0.044715f * powf(x, 3.f))));
                uint64_t index = (int)((x - range_start) * fit_split);
                float c2 = fast_gelu_poly_table_float[index][0];
                float c1 = fast_gelu_poly_table_float[index][1];
                float c0 = fast_gelu_poly_table_float[index][2];
                data_ldm[i] = ((c2 * x) + c1) * x + c0;
            }
        }
        CRTS_dma_put(data + bs, data_ldm, bl * sizeof(float));
    }
}