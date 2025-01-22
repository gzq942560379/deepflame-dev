#pragma once
#include "lduMatrix.H"
#include <cassert>
#include "messageStream.H"

namespace Foam{

enum InnerMatrixFormat{
    DFMATRIX_INNERMATRIX_FORMAT_LDU,
    DFMATRIX_INNERMATRIX_FORMAT_CSR,
    DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR,
};

class dfInnerMatrix{
protected:
    label n_;
    scalarField diag_;
    bool diagonal_;
    bool symmetric_;
public:
    dfInnerMatrix(const lduMatrix& ldu):
        diagonal_(ldu.diagonal()),
        symmetric_(ldu.symmetric())
    {
        assert(ldu.hasDiag());
        if(ldu.hasDiag())
        {
            diag_ = ldu.diag();
            n_ = ldu.diag().size();
        }
    };
    virtual ~dfInnerMatrix(){};

    const label& n() const {return n_;};
    scalarField& diag(){return diag_;};
    const scalarField& diag() const {return diag_;};
    bool diagonal() const {return diagonal_;};
    bool symmetric() const {return symmetric_;};
    bool asymmetric() const {return !symmetric_;};

    virtual InnerMatrixFormat getFormat() const = 0;
    virtual const std::string getFormatName() const = 0;

    virtual void SpMV
    (
        scalar* const __restrict__ ApsiPtr,
        const scalar* const __restrict__ psiPtr
    ) const = 0;
    virtual void GaussSeidel
    (
        scalar* const __restrict__ psiPtr,
        scalar* const __restrict__ bPrimePtr
    ) const = 0;
    virtual void Jacobi
    (
        scalar* const __restrict__ psiPtr,
        scalar* const __restrict__ bPrimePtr
    ) const = 0;
    virtual void calcDILUReciprocalD(
        scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void DILUPrecondition(
        scalar* const __restrict__ wAPtr,
        const scalar* const __restrict__ rAPtr,
        const scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void DILUPreconditionT(
        scalar* const __restrict__ wTPtr,
        const scalar* const __restrict__ rTPtr,
        const scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void calcDICReciprocalD(
        scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void DICPrecondition(
        scalar* const __restrict__ wAPtr,
        const scalar* const __restrict__ rAPtr,
        const scalar* const __restrict__ rDPtr
    ) const = 0;
};

} // End namespace Foam

