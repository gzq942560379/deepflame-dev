#include "BCSR.H"
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <cstdio>

namespace Foam{

BCSR::BCSR(const csrPattern csr, const labelList& regionPtr){

    n_ = csr.n();
    rowBlockCount_ = regionPtr.size() - 1; 
    rowBlockPtr_ = regionPtr;

    std::vector<std::vector<std::pair<label, label>>> blocks(rowBlockCount_ * rowBlockCount_);

    const auto& csrRowPtr = csr.rowptr();
    const auto& csrColIdx = csr.colidx();

    for(label r = 0; r < csr.n(); ++r){
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        assert(rbid < rowBlockCount_);
        assert(rbid >= 0);
        for(label idx = csrRowPtr[r]; idx < csrRowPtr[r+1]; ++idx){
            label c = csrColIdx[idx];
            label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
            assert(cbid < rowBlockCount_);
            assert(cbid >= 0);
            blocks[bid2d(rbid, cbid)].push_back({r,c});
        }
    }


    int mpirank, mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    if(mpirank == 0){
        Info << "Each Block Info : " << endl;
        for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
            for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
                // Info << setw(12) << setfill(' ') << blocks[bid2d(rbid, cbid)].size() << "\t";
                printf("%12d\t", blocks[bid2d(rbid, cbid)].size());
            }
            printf("\n");
            fflush(stdout);
        }
    }

}

}
