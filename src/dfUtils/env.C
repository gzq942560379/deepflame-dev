#include "env.H"
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>


bool env_get_bool(const char* name, bool default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    if(std::strcmp(tmp, "true") == 0){
        return true;
    }else{
        return false;
    }
}

int env_get_int(const char* name, int default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    return std::atoi(tmp);
}

int dnn_batch_size = env_get_int("DNN_BATCH_SIZE", 131072);

void env_show(){
    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    if(mpirank != 0) return;
    
    std::cout << std::endl;
    std::cout << "env show --------------------------------" << std::endl;
    
    std::cout << "dnn_batch_size : " << dnn_batch_size << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
}

