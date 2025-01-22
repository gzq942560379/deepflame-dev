#include "dfBlockMatrix.H"
#include <cassert>
#include <mpi.h>
#include "dfCSRSubMatrix.H"
#include "env.H"

namespace Foam{

dfBlockMatrix::dfBlockMatrix(const lduMatrix& ldu, const labelList& rowBlockPtr):dfInnerMatrix(ldu),rowBlockCount_(rowBlockPtr.size()-1),rowBlockPtr_(rowBlockPtr){
    // Pout << "Enter dfBlockMatrix::dfBlockMatrix(lduMatrix& ldu, const labelList& rowBlockPtr)" << endl << flush;
    blocks_.resize(rowBlockCount_ * rowBlockCount_);
    // off_diagonal_nnz_ = 0;
    if(!ldu.hasLower() && !ldu.hasUpper()){
        return;
    }

    maxBlockSize_ = 0;
    for(label bid = 0; bid < rowBlockCount_; ++bid){
        maxBlockSize_ = std::max(maxBlockSize_, rowBlockPtr_[bid + 1] - rowBlockPtr_[bid]);
    }

    std::vector<std::vector<std::tuple<label,label,scalar>>> blocksTmp(rowBlockCount_ * rowBlockCount_);

    const labelList& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const labelList& lduUpperAddr = ldu.lduAddr().upperAddr();
    // lower[i] (lduUpperAddr[i], lduLowerAddr[i]) 
    const scalarList& lduLower = ldu.lower();
    // upper[i] (lduLowerAddr[i], lduUpperAddr[i])
    const scalarList& lduUpper = ldu.upper();

    // lower
    // (lduUpperAddr[i], lduLowerAddr[i])
    for(label i = 0; i < lduLower.size(); ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        scalar v = lduLower[i];
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
        blocksTmp[bid2d(rbid, cbid)].push_back({r,c,v});
    }

    // upper
    // (lduLowerAddr[i], lduUpperAddr[i])
    for(label i = 0; i < lduUpper.size(); ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        scalar v = lduUpper[i];
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
        blocksTmp[bid2d(rbid, cbid)].push_back({r,c,v});
    }

    // convert each block to CSR
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            const std::vector<std::tuple<label,label,scalar>>& block = blocksTmp[bid];
            if(block.size() == 0){
                continue;
            }
            
            label rowStart = rowBlockPtr_[rbid];
            label rowEnd = rowBlockPtr_[rbid + 1];
            label rowLen = rowEnd - rowStart;
            label colStart = rowBlockPtr_[cbid];
            label colEnd = rowBlockPtr_[cbid + 1];
            label colLen = colEnd - colStart;

            // count nnz per row
            std::vector<label> nnzPerRow(rowLen, 0);
            
            for(const auto& entry: block){
                label r = std::get<0>(entry);
                nnzPerRow[r - rowStart] += 1;
            }

            std::unique_ptr<label[]> rowPtr = std::make_unique<label[]>(rowLen + 1);
            std::vector<label> curIndexPerRow(rowLen + 1);

            rowPtr[0] = 0;
            curIndexPerRow[0] = 0;
            for(label i = 0; i < rowLen; ++i){
                rowPtr[i + 1] = rowPtr[i] + nnzPerRow[i];
                curIndexPerRow[i + 1] = rowPtr[i + 1];
            }

            label nnz_block = rowPtr[rowLen];

            std::unique_ptr<label[]> colIdx = std::make_unique<label[]>(nnz_block);
            std::unique_ptr<scalar[]> values = std::make_unique<scalar[]>(nnz_block);

            for(const auto& entry: block){
                label r = std::get<0>(entry);
                label c = std::get<1>(entry);
                scalar v = std::get<2>(entry);
                label rowIdx = r - rowStart;
                label idx = curIndexPerRow[rowIdx];
                colIdx[idx] = c - colStart;
                values[idx] = v;
                curIndexPerRow[rowIdx] += 1;
            }

            blocks_[bid] = std::make_unique<dfCSRSubMatrix>(rowLen, colLen, rowPtr.release(), colIdx.release(), values.release());
        }
    }
    // Pout << "Exit dfBlockMatrix::dfBlockMatrix(lduMatrix& ldu, const labelList& rowBlockPtr)" << endl << flush;
}


void dfBlockMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr) const {
    // Pout << "Enter dfBlockMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid + 1] - rowBlockPtr_[rbid];
        scalar* const __restrict__ ApsiPtr_offset = ApsiPtr + rowOffset;
        for(label r = 0; r < rowLen; ++r){
            ApsiPtr_offset[r] = diagPtr[r] * psiPtr[r];
        }
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(blocks_[bid] == nullptr){
                continue;
            }
            const dfBlockSubMatrix& csrSubMatrix = *blocks_[bid];
            label colOffset = rowBlockPtr_[cbid];
            csrSubMatrix.SpMV(ApsiPtr_offset, psiPtr + colOffset);
        }
    }

    // Pout << "Exit dfBlockMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr)" << endl << flush;
}

void dfBlockMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    // Pout << "Enter dfBlockMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid+1] - rowBlockPtr_[rbid];
        scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rowOffset;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;

        // B = b - (L + U) * x
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(rbid == cbid)
                continue;
            if(blocks_[bid] == nullptr)
                continue;
            label colOffset = rowBlockPtr_[cbid];
            const dfBlockSubMatrix& offDiagBlock = *blocks_[bid];
            offDiagBlock.BsubApsi(bPrimePtr_offset, psiPtr + colOffset);
        }
        label diagBlockIndex = bid2d(rbid, rbid);
        scalar* const __restrict__ psiPtr_offset = psiPtr + rowOffset;
        if(blocks_[diagBlockIndex] == nullptr){
            //  x = B / diag
            for(label r = 0; r < rowLen; ++r){
                psiPtr_offset[r] = bPrimePtr_offset[r] / diagPtr_offset[r];
            }
        }else{
            const dfBlockSubMatrix& diagBlock = *blocks_[diagBlockIndex];
            diagBlock.GaussSeidel(psiPtr_offset, bPrimePtr_offset, diagPtr_offset);
        }

    }
    // Pout << "Exit dfBlockMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
}

void dfBlockMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    // Pout << "Enter dfBlockMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    std::unique_ptr<scalar[]> psiOldPtr = std::make_unique<scalar[]>(n_);

    for(label r = 0; r < n_; ++r){
        psiOldPtr[r] = psiPtr[r];
    }

    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid+1] - rowBlockPtr_[rbid];
        scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rowOffset;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;
        // b = b - (L + U) * x
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(blocks_[bid] == nullptr){
                continue;
            }
            label colOffset = rowBlockPtr_[cbid];
            const dfBlockSubMatrix& offDiagBlock = *blocks_[bid];
            offDiagBlock.BsubApsi(bPrimePtr_offset, psiOldPtr.get() + colOffset);
        }
        // x = B / diag
        scalar* const __restrict__ psiPtr_offset = psiPtr + rowOffset;
        for(label r = 0; r < rowLen; ++r){
            psiPtr_offset[r] = bPrimePtr_offset[r] / diagPtr_offset[r];
        }
    }

    // Pout << "Exit dfBlockMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
}

void dfBlockMatrix::calcDILUReciprocalD(scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::DILUPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::DILUPreconditionT(scalar* const __restrict__ wTPtr, const scalar* const __restrict__ rTPtr, const scalar* const __restrict__ rDPtr) const{
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::calcDICReciprocalD(scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::DICPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

} // End namespace Foam
