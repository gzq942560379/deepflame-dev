#!/bin/sh

print_finish() {
    if [ ! -z "$LIBTORCH_ROOT" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and libtorch) compiled successfully! Enjoy!! |"
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        return
    fi
    if [ ! -z "$USE_BLASDNN" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and blasdnn) compiled successfully! Enjoy!! | "
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
	return
    fi
    if [ ! -z "$PYTHON_LIB_DIR" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and pytorch) compiled successfully! Enjoy!! | "
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
	return
    fi
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
    echo "| deepflame (linked with libcantera) compiled successfully! Enjoy!! |"
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
}
if [ ! -z "$LIBTORCH_ROOT"  ]; then
    cd "$DF_SRC/dfChemistryModel/DNNInferencer"
    mkdir build
    cd build
    # # x86_64
    # cmake ..
    # fugaku
    cmake .. -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_CXX_FLAGS="-Nclang -Ofast -g -mlittle-endian" -DLIBTENSORFLOW_ROOT=$LIBTENSORFLOW_ROOT
    cp ./libDNNInferencer.so $DF_ROOT/lib/
fi
if [ ! -z "$USE_BLASDNN" ]; then
    cd "$DF_SRC/dfChemistryModel/DNNInferencer_blas"
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_CXX_FLAGS="-Nclang -Ofast -g -fopenmp -Nfjomplib -std=c++11"
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    make VERBOSE=1
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    cp ./libDNNInferencer_blas.so $DF_ROOT/lib/
fi
cd $DF_ROOT
./Allwmake -j && print_finish
