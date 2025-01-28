#pragma once
#include "dfInnerMatrix.H"
#include "dfBlockSubMatrix.H"
#include "dfCSRSubMatrix.H"
#include <vector>

#define bid2d(rbid, cbid) (rbid * rowBlockCount_ + cbid)

namespace Foam{

class dfBlockMatrix : public dfInnerMatrix{
private:

    label rowBlockCount_;
    labelList rowBlockPtr_;
    std::vector<std::unique_ptr<dfBlockSubMatrix>> blocks_;

    void buildBlocks(const lduMatrix& ldu);

public:
    dfBlockMatrix(const lduMatrix& ldu, const labelList& rowBlockPtr);

    dfBlockMatrix(const lduMatrix& courseLduMatrix, const labelList& fineRowBlockPtr, const labelList& fineToCoarse);

    ~dfBlockMatrix(){};

    const labelList& rowBlockPtr() const {return rowBlockPtr_;};

    InnerMatrixFormat getFormat() const {return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR;};
    const std::string getFormatName() const {return "BLOCK_CSR";};

    void SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr) const;
    
    void GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const;
    void Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const;

    void calcDILUReciprocalD(scalar* const __restrict__ rDPtr) const;
    void DILUPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const;
    void DILUPreconditionT(scalar* const __restrict__ wTPtr, const scalar* const __restrict__ rTPtr, const scalar* const __restrict__ rDPtr) const;

    void calcDICReciprocalD(scalar* const __restrict__ rDPtr) const;
    void DICPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const;
};

}