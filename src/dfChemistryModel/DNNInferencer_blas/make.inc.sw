

YAML_HOME   =   /home/export/online1/mdt00/shisuan/jiaweile/guozhuoqiang/DeepFlame/software/yaml-cpp-0.7.0
YAML_INC    =   -I$(YAML_HOME)/include
YAML_LIB    =   -L$(YAML_HOME)/lib64 -lyaml-cpp

BLAS_LIB    =   -L/home/export/online1/mdt00/shisuan/jiaweile/guozhuoqiang/DeepFlame/software/xMath-SACA -lswblas

MATH_LIB    =   -lm -lm_slave

DEF		=	

INC     =   $(YAML_INC)
LIB     =   $(YAML_LIB) $(BLAS_LIB) $(MATH_LIB)

CC      =   mpicc
CXX     =   mpicxx
AR      =   swar
ARFLAGS =   cr
RANLIB  =   swranlib

CFLAGS      =   -O3 -msimd -g -mftz -mieee $(DEF) $(INC)
CXXFLAGS    =   -O3 -msimd -g -mftz -mieee $(DEF) $(INC)
LINKFLAGS   =   -mhybrid
