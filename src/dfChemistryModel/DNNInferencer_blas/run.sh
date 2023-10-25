#!/bin/bash


rm gelu.dat
bsub -J geluf_test -q q_share -n 1 -share_size 15536  -b -cgsp 64  -o gelu.dat ./geluf_test 