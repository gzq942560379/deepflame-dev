#include "memory.H"

void* df_alloc(size_t size){
#ifdef __sw_64__
    return libc_aligned_malloc(size);
#else
    return aligned_alloc(64, size);
#endif
}

void df_free(void* ptr){
#ifdef __sw_64__
    return libc_aligned_free(ptr);
#else
    return free(ptr);
#endif
}