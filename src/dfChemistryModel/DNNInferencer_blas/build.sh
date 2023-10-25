#!/bin/bash

rm -rf *.o *.a gelu_s_test

YAML_HOME="/home/export/online1/mdt00/shisuan/jiaweile/guozhuoqiang/DeepFlame/software/yaml-cpp-0.7.0"
YAML_INC="-I$YAML_HOME/include"
YAML_LIB="-L$YAML_HOME/lib64 -lyaml-cpp"

BLAS_LIB="-L/home/export/online1/mdt00/shisuan/jiaweile/guozhuoqiang/DeepFlame/software/xMath-SACA -lswblas"

MATH_LIB="-lm -lm_slave"

INC="$YAML_INC"
LIB="$YAML_LIB $BLAS_LIB $MATH_LIB"

# DEF="-DDEF_PROFILING"
DEF=""

CC="mpicc"
CXX="mpicxx"
AR="swar"
ARFLAGS="cr"
RANLIB="swranlib"
CFLAGS="-O3 -msimd -g -mftz -mieee $DEF $INC"
CXXFLAGS="-O3 -msimd -g -mftz -mieee $DEF $INC"
LINKFLAGS="-mhybrid"

set -ex

$CXX -mhost $CXXFLAGS -c gelu_s_test.cpp -o gelu_s_test.o
$CC -mslave $CFLAGS -c gelu_s_naive.slave.c -o gelu_s_naive.slave.o
$CC -mslave $CFLAGS -c gelu_s_exp.slave.c -o gelu_s_exp.slave.o
$CC -mslave $CFLAGS -c gelu_s_ldm.slave.c -o gelu_s_ldm.slave.o
$CC -mslave $CFLAGS -c gelu_s_ldm_lookup.slave.c -o gelu_s_ldm_lookup.slave.o

$CC -mslave $CFLAGS -c bias_s_naive.slave.c -o bias_s_naive.slave.o
$CC -mslave $CFLAGS -c bias_s_ldm.slave.c -o bias_s_ldm.slave.o

$CC -mslave $CFLAGS -c bias_gelu_s_ldm_lookup.slave.c -o bias_gelu_s_ldm_lookup.slave.o

$CXX $LINKFLAGS gelu_s_test.o gelu_s_naive.slave.o -o gelu_s_test $LIB

$CXX -mhost $CXXFLAGS -c DNNInferencer_blas.cpp -o DNNInferencer_blas.o
$CXX -mhost $CXXFLAGS -c Layer.cpp -o Layer.o

$AR $ARFLAGS libDNNInferencer_blas.a DNNInferencer_blas.o Layer.o \
    gelu_s_naive.slave.o gelu_s_exp.slave.o gelu_s_ldm.slave.o gelu_s_ldm_lookup.slave.o \
    bias_s_naive.slave.o bias_s_ldm.slave.o bias_gelu_s_ldm_lookup.slave.o
$RANLIB libDNNInferencer_blas.a