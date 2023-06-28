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
    mutable double solve_time = 0.;
    mutable double spmv_time = 0.;
    mutable double normFactor_time = 0.;
    mutable double gSumMag_time = 0.;
    mutable double gSumProd_time = 0.;
    mutable double localUpdate_time = 0.;
    mutable double precondition_time = 0.;
    mutable double precondition_construct_time = 0.;

    void print_time() const {
        double other_time = solve_time - spmv_time - normFactor_time - gSumMag_time - gSumProd_time - localUpdate_time - precondition_time;
        Info << "DIVPCG solve profiling -----------------------------------------------------------------" << endl;
        Info << "solve time : " << solve_time << endl;
        Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
        Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
        Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
        Info << "gSumProd time : " << gSumProd_time << ", " << gSumProd_time / solve_time * 100 << "%" << endl;
        Info << "localUpdate time : " << localUpdate_time << ", " << localUpdate_time / solve_time * 100 << "%" << endl;
        Info << "precondition time : " << precondition_time << ", " << precondition_time / solve_time * 100 << "%" << endl;
        Info << "other time : " << other_time << ", " << other_time / solve_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();
    }
};

}

