#pragma once
#include "dfMatrix.H"


namespace Foam{

class dfLduMatrix : public dfMatrix::InnerMatrix{
private:
    // - Coefficients (not including interfaces)
    scalarField lower_, upper_;
    labelList lowerAddr_;
    labelList upperAddr_;
    labelList ownerStartAddr_;
    labelList losortAddr_;

public:
    dfLduMatrix(const lduMatrix& ldu);

    ~dfLduMatrix(){};

    scalarField& lower() {return symmetric_ ? upper_ : lower_;};
    scalarField& upper() {return upper_;};
    const scalarField& lower() const {return symmetric_ ? upper_ : lower_;};
    const scalarField& upper() const {return upper_;};
    labelList& lowerAddr() {return lowerAddr_;};
    labelList& upperAddr() {return upperAddr_;};
    const labelList& lowerAddr() const {return lowerAddr_;};
    const labelList& upperAddr() const {return upperAddr_;};
    labelList& ownerStartAddr() {return ownerStartAddr_;};
    const labelList& ownerStartAddr() const {return ownerStartAddr_;};
    labelList& losortAddr() {return losortAddr_;};
    const labelList& losortAddr() const {return losortAddr_;};

    void SpMV(scalarField& Apsi, const scalarField& psi) const;
    
    void GaussSeidel(scalarField& psi, scalarField& bPrime) const;

    void calcDILUReciprocalD(scalarField& rD) const;
    void DILUPrecondition(scalarField& wA, const scalarField& rA, const scalarField& rD) const;
    void DILUPreconditionT(scalarField& wT, const scalarField& rT, const scalarField& rD) const;

    void calcDICReciprocalD(scalarField& rD) const;
    void DICPrecondition(scalarField& wA, const scalarField& rA, const scalarField& rD) const;
    
};

}