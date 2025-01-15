#include "dfLduMatrix.H"
#include <cassert>

namespace Foam{

dfLduMatrix::dfLduMatrix(const lduMatrix& ldu):dfMatrix::InnerMatrix(ldu){
    lowerAddr_ = ldu.lduAddr().lowerAddr();
    upperAddr_ = ldu.lduAddr().upperAddr();
    ownerStartAddr_ = ldu.lduAddr().ownerStartAddr();
    losortAddr_ = ldu.lduAddr().losortAddr();
    if(ldu.hasLower())
    {
        lower_ = ldu.lower();
    }
    if(ldu.hasUpper())
    {
        upper_ = ldu.upper();
    }
}


void dfLduMatrix::SpMV(scalarField& Apsi, const scalarField& psi) const {
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_.begin();

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    const label nCells = diag_.size();
    for (label cell=0; cell<nCells; cell++)
    {
        ApsiPtr[cell] = diagPtr[cell]*psiPtr[cell];
    }

    const label nFaces = upper().size();

    for (label face=0; face<nFaces; face++)
    {
        ApsiPtr[uPtr[face]] += lowerPtr[face]*psiPtr[lPtr[face]];
        ApsiPtr[lPtr[face]] += upperPtr[face]*psiPtr[uPtr[face]];
    }
}

void dfLduMatrix::GaussSeidel(scalarField& psi, scalarField& bPrime) const {
    const label nCells = psi.size();
    scalar* __restrict__ psiPtr = psi.begin();
    scalar* __restrict__ bPrimePtr = bPrime.begin();

    const scalar* const __restrict__ diagPtr = diag_.begin();
    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();
    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ ownStartPtr = ownerStartAddr().begin();

    scalar psii;
    label fStart;
    label fEnd = ownStartPtr[0];

    for (label celli=0; celli<nCells; celli++)
    {   
        // Start and end of this row
        fStart = fEnd;
        fEnd = ownStartPtr[celli + 1];
        // Get the accumulated neighbour side
        psii = bPrimePtr[celli];

        // Accumulate the owner product side
        for (label facei=fStart; facei<fEnd; facei++)
        {
            psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
        }

        // Finish psi for this cell
        psii /= diagPtr[celli];

        // Distribute the neighbour side using psi for this cell
        for (label facei=fStart; facei<fEnd; facei++)
        {
            bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
        }

        psiPtr[celli] = psii;
    }
}

void dfLduMatrix::calcDILUReciprocalD(scalarField& rD) const {
    assert(this->asymmetric());

    scalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nFaces = upper().size();

    for (label face=0; face<nFaces; face++)
    {
        rDPtr[uPtr[face]] -= upperPtr[face]*lowerPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    label nCells = rD.size();

    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/rDPtr[cell];
    }
}

void dfLduMatrix::DILUPrecondition(scalarField& wA, const scalarField& rA, const scalarField& rD) const {
    assert(this->asymmetric());
    scalar* __restrict__ wAPtr = wA.begin();
    const scalar* __restrict__ rAPtr = rA.begin();
    const scalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const label* const __restrict__ losortPtr = losortAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nCells = wA.size();
    label nFaces = upper().size();
    label nFacesM1 = nFaces - 1;

    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    }

    label sface;

    for (label face=0; face<nFaces; face++)
    {
        sface = losortPtr[face];
        wAPtr[uPtr[sface]] -=
            rDPtr[uPtr[sface]]*lowerPtr[sface]*wAPtr[lPtr[sface]];
    }

    for (label face=nFacesM1; face>=0; face--)
    {
        wAPtr[lPtr[face]] -=
            rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    }
}

void dfLduMatrix::DILUPreconditionT(scalarField& wT, const scalarField& rT, const scalarField& rD) const{
    assert(this->asymmetric());
    scalar* __restrict__ wTPtr = wT.begin();
    const scalar* __restrict__ rTPtr = rT.begin();
    const scalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const label* const __restrict__ losortPtr = losortAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nCells = wT.size();
    label nFaces = upper().size();
    label nFacesM1 = nFaces - 1;

    for (label cell=0; cell<nCells; cell++)
    {
        wTPtr[cell] = rDPtr[cell]*rTPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        wTPtr[uPtr[face]] -=
            rDPtr[uPtr[face]]*upperPtr[face]*wTPtr[lPtr[face]];
    }

    label sface;

    for (label face=nFacesM1; face>=0; face--)
    {
        sface = losortPtr[face];
        wTPtr[lPtr[sface]] -=
            rDPtr[lPtr[sface]]*lowerPtr[sface]*wTPtr[uPtr[sface]];
    }
}

void dfLduMatrix::calcDICReciprocalD(scalarField& rD) const {
    assert(this->symmetric());

    scalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();

    // Calculate the DIC diagonal
    const label nFaces = upper().size();
    for (label face=0; face<nFaces; face++)
    {
        rDPtr[uPtr[face]] -= upperPtr[face]*upperPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    const label nCells = rD.size();

    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/rDPtr[cell];
    }
}

void dfLduMatrix::DICPrecondition(scalarField& wA, const scalarField& rA, const scalarField& rD) const {
    assert(this->symmetric());

    scalar* __restrict__ wAPtr = wA.begin();
    const scalar* __restrict__ rAPtr = rA.begin();
    const scalar* __restrict__ rDPtr = rD.begin();

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();

    label nCells = wA.size();
    label nFaces = upper().size();
    label nFacesM1 = nFaces - 1;

    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        wAPtr[uPtr[face]] -= rDPtr[uPtr[face]]*upperPtr[face]*wAPtr[lPtr[face]];
    }

    for (label face=nFacesM1; face>=0; face--)
    {
        wAPtr[lPtr[face]] -= rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    }
}

} // End namespace Foam
