#include <stdlib.h>
#include <stdio.h>
#include <crts.h>
#include <math.h>
#include "slave_kernel.h"

int main(){
    
    CRTS_init();
    float range_start = -20.;
    float range_end = 20.;

    int64_t len = (range_end - range_start) * 100;
    float* input = (float*)malloc(len * sizeof(float));
    float* output = (float*)malloc(len * sizeof(float));
    for(int64_t i = 0; i < len; ++i){
        input[i] = 1.0 * i / len * (range_end - range_start) + range_start;
        output[i] = 1.0 * i / len * (range_end - range_start) + range_start;
    }

    gelu_s_param_t para;
    para.len = len;
    para.data = output;

    CRTS_athread_spawn(reinterpret_cast<void *>(SLAVE_FUN(gelu_s_naive)), &para);
    CRTS_athread_join();

    for(int64_t i = 0; i < len; ++i){
        float diff = input[i] < 0. ? fabsf(output[i]) : fabsf(input[i] - output[i]);
        printf("%12.8lf %12.8lf %12.8lf\n", input[i], output[i], diff);
    }

    return 0;
}
