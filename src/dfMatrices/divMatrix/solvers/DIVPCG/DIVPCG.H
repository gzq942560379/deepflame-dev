#pragma once

#include "divMatrix.H"

namespace Foam
{

class DIVPCG: public  divMatrix::solver{

    autoPtr<divMatrix::preconditioner> preconPtr_;


public:
    //- Runtime type information
    TypeName("PCG");

    //- Construct from matrix components and solver controls
    DIVPCG
    (
        const word& fieldName,
        const divMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    DIVPCG(const DIVPCG&) = delete;

    //- Destructor
    virtual ~DIVPCG(){}

    // Member Functions

    //- Solve the matrix with this solver
    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt=0
    ) const;


    // Member Operators

    //- Disallow default bitwise assignment
    void operator=(const DIVPCG&) = delete;


    // profiling
    mutable double PCG_time = 0.;
    mutable double PCG_spmv_time = 0.;
    mutable double PCG_normFactor_time = 0.;
    mutable double PCG_gSumMag_time = 0.;
    mutable double PCG_gSumProd_time = 0.;
    mutable double PCG_localUpdate_time = 0.;
    mutable double PCG_precondition_time = 0.;
    mutable double PCG_misc_time = 0.;

    void print_time() const {
        double PCG_other_time = PCG_time - PCG_spmv_time - PCG_normFactor_time - PCG_gSumMag_time - PCG_gSumProd_time - PCG_localUpdate_time - PCG_precondition_time;
        Info << "DIVPCG solve profiling -----------------------------------------------------------------" << endl;
        Info << "PCG time : " << PCG_time << endl;
        Info << "PCG spmv time : " << PCG_spmv_time << ", " << PCG_spmv_time / PCG_time * 100 << "%" << endl;
        Info << "PCG normFactor time : " << PCG_normFactor_time << ", " << PCG_normFactor_time / PCG_time * 100 << "%" << endl;
        Info << "PCG gSumMag time : " << PCG_gSumMag_time << ", " << PCG_gSumMag_time / PCG_time * 100 << "%" << endl;
        Info << "PCG gSumProd time : " << PCG_gSumProd_time << ", " << PCG_gSumProd_time / PCG_time * 100 << "%" << endl;
        Info << "PCG localUpdate time : " << PCG_localUpdate_time << ", " << PCG_localUpdate_time / PCG_time * 100 << "%" << endl;
        Info << "PCG precondition time : " << PCG_precondition_time << ", " << PCG_precondition_time / PCG_time * 100 << "%" << endl;
        Info << "PCG other time : " << PCG_other_time << ", " << PCG_other_time / PCG_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();

    }
};

}

