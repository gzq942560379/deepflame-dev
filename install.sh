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
    cmake .. -DCMAKE_CXX_COMPILER=mpiFCC -DCMAKE_CXX_FLAGS="-Nclang -Ofast -g -mlittle-endian" -DLIBTENSORFLOW_ROOT=$LIBTENSORFLOW_ROOT
    cp ./libDNNInferencer.so $DF_ROOT/lib/
fi
if [ ! -z "$USE_BLASDNN" ]; then
    cd "$DF_SRC/dfChemistryModel/DNNInferencer_blas"
    # mkdir build
    # cd build
    # cmake .. -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_AR=/usr/sw/swgcc/swgcc710-tools-SEA-1307/usr/bin/swar -DCMAKE_CXX_FLAGS="-O3 -msimd -g -mftz -mieee" -DCMAKE_EXE_LINKER_FLAGS="-mhybrid" -DBLAS_LIBRARY="-L/home/export/online1/mdt00/shisuan/jiaweile/guozhuoqiang/DeepFlame/software/xMath-SACA -lswblas"
    # make VERBOSE=1
    # mkdir -p $DF_ROOT/lib/
    # cp ./libDNNInferencer_blas.a $DF_ROOT/lib/
    make -j4
    cp ./libDNNInferencer_blas.a $DF_ROOT/lib/

fi
cd $DF_ROOT

# ./Allwclean
./Allwmake -j && print_finish
