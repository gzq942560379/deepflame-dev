#include <crts.h> 
#include <simd.h>
#include <math.h>
#include "slave_param.h"

#define MAX_BIAS 3200

void bias_s_naive(bias_s_param_t *para_p){
    float* input = para_p->input;
    float* bias = para_p->bias;
    int64_t row = para_p->row;
    int64_t col = para_p->col;

    int64_t row_start = CRTS_tid * row / CRTS_MAX_SPE_NUM;
    int64_t row_end = (CRTS_tid + 1) * row / CRTS_MAX_SPE_NUM;


    for(int64_t r = row_start; r < row_end; ++r){
        float* input_row = input + r * col;
        for(int64_t c = 0; c < col; ++c){
            input_row[c] += bias[c];
        }
    }
}