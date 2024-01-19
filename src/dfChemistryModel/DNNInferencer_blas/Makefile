
include  make.inc

LIB = libDNNInferencer_blas.a

.PHONY : all lib clean

INC += -I./include

HOST_CPP_SRC = $(shell find ./host -name "*.cpp")
HOST_C_SRC = $(shell find ./host -name "*.c")
SLAVE_SRC = $(shell find ./slave -name "*.c")

HOST_CPP_OBJ = $(patsubst %.cpp, %.o, $(HOST_CPP_SRC))
HOST_C_OBJ = $(patsubst %.c, %.o, $(HOST_C_SRC))
SLAVE_OBJ = $(patsubst %.c, %.o, $(SLAVE_SRC))

all : lib

lib: $(HOST_CPP_OBJ) $(HOST_C_OBJ) $(SLAVE_OBJ)
	$(AR) $(ARFLAGS) $(LIB) $^
	$(RANLIB) $(LIB)

$(HOST_CPP_OBJ) : %.o : %.cpp
	$(CXX) -mhost $(CXXFLAGS) -c $< -o $@

$(HOST_C_OBJ) : %.o : %.c
	$(CXX) -mhost $(CXXFLAGS) -c $< -o $@

$(SLAVE_OBJ) : %.o : %.c
	$(CC) -mslave $(CFLAGS) -c $< -o $@

clean : 
	rm -rf $(HOST_CPP_OBJ) $(HOST_C_OBJ) $(SLAVE_OBJ) $(LIB)

