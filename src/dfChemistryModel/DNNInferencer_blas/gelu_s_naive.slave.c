#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"

void gelu_s_naive(gelu_s_param_t *para_p){
    int64_t len = para_p->len;
    float* data = para_p->data;

    int64_t start = CRTS_tid * len / CRTS_MAX_SPE_NUM;
    int64_t end = (CRTS_tid + 1) * len / CRTS_MAX_SPE_NUM;

    for(int64_t i = start; i < end; ++i){
        float x = data[i];
        data[i] = 0.5 * x * (1.f + tanhf(sqrtf(2.f / M_PI) * (x + 0.044715f * powf(x, 3.f))));
    }

}