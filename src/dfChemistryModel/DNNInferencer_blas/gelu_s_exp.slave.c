#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"

void gelu_s_exp(gelu_s_param_t *para_p){
    const double const_1 = sqrtf(2. / M_PI);
    const double const_2 = 0.044715;
    const double one = 1.;
    const double half = 0.5;

    int64_t len = para_p->len;
    float* data = para_p->data;

    int64_t start = CRTS_tid * len / CRTS_MAX_SPE_NUM;
    int64_t end = (CRTS_tid + 1) * len / CRTS_MAX_SPE_NUM;

    for(int64_t i = start; i < end; ++i){
        float x = data[i];
        float tmp = const_1 * (x + const_2 * x * x * x);
        float tanh = 1.f - 2.f / (expf(2.f * tmp) + 1.f);
        data[i] = half * x * (one + tanh);
    }
}