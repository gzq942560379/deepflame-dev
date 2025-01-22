#pragma once

#include "dfBlockSubMatrix.H"
#include <memory>


namespace Foam{

class dfCSRSubMatrix : public dfBlockSubMatrix{
private:

    std::unique_ptr<label> rowPtr_;
    std::unique_ptr<label> colIdx_;
    std::unique_ptr<scalar> values_;

public:
    dfCSRSubMatrix(label nRows, label nCols, label* rowPtr, label* colIdx, scalar* values):dfBlockSubMatrix(nRows, nCols),rowPtr_(rowPtr),colIdx_(colIdx),values_(values){}
    ~dfCSRSubMatrix(){}

    void SpMV(
        scalar* const __restrict__ ApsiPtr_offset,
        const scalar* const __restrict__ psiPtr_offset
    ) const;

    void BsubApsi(
        scalar* const __restrict__ BPtr_offset,
        const scalar* const __restrict__ psiPtr_offset
    ) const;

    void GaussSeidel(
        scalar* const __restrict__ psiPtr_offset,
        const scalar* const __restrict__ BPtr_offset,
        const scalar* const __restrict__ diagPtr_offset
    ) const;

    // void Jacobi(
    //     scalar* const __restrict__ psiPtr_offset,
    //     const scalar* const __restrict__ psiOldPtr_offset,
    //     const scalar* const __restrict__ BPtr_offset,
    //     const scalar* const __restrict__ diagPtr_offset
    // ) const;
};

}
