#include "dfCSRSubMatrix.H"

namespace Foam{

void dfCSRSubMatrix::SpMV(scalar* const __restrict__ ApsiPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_.get();
    const scalar* const __restrict__ valuePtr = values_.get();

    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.;
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            sum += valuePtr[idx] * psiPtr_offset[colIdxPtr[idx]];
        }
        ApsiPtr_offset[r] += sum;
    }
}


void dfCSRSubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {
    // Info << "Enter dfCSRSubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset)" << endl << flush; 
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_.get();
    const scalar* const __restrict__ valuePtr = values_.get();
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.0;
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            sum += valuePtr[idx] * psiPtr_offset[colIdxPtr[idx]];
        }
        BPtr_offset[r] -= sum;
    }
    // Info << "Exit dfCSRSubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset)" << endl << flush; 
}

void dfCSRSubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset) const {
    // Info << "Enter dfCSRSubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_.get();
    const scalar* const __restrict__ valuePtr = values_.get();
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.0;
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            sum += valuePtr[idx] * psiPtr_offset[colIdxPtr[idx]];
        }
        psiPtr_offset[r] = (BPtr_offset[r] - sum) / diagPtr_offset[r];
    }
    // Info << "Exit dfCSRSubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
}

// void dfCSRSubMatrix::Jacobi(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ psiOldPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset) const {
//     Info << "Enter dfCSRSubMatrix::Jacobi(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ psiOldPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
//     const label* const __restrict__ rowPtr = rowPtr_.get();
//     const label* const __restrict__ colIdxPtr = colIdx_.get();
//     const scalar* const __restrict__ valuePtr = values_.get();
//     for(label r = 0; r < nRows_; ++r){
//         scalar sum = 0.0;
//         for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
//             sum += valuePtr[idx] * psiOldPtr_offset[colIdx_[idx]];
//         }
//         psiPtr_offset[r] = (BPtr_offset[r] - sum) / diagPtr_offset[r];
//     }
//     Info << "Exit dfCSRSubMatrix::Jacobi(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ psiOldPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
// }

}
