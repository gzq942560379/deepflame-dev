#pragma once

#include "csrMatrix.H"

namespace Foam
{

class CSRPCG: public  csrMatrix::solver{
public:
    //- Runtime type information
    TypeName("PCG");

    //- Construct from matrix components and solver controls
    CSRPCG
    (
        const word& fieldName,
        const csrMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    CSRPCG(const CSRPCG&) = delete;

    //- Destructor
    virtual ~CSRPCG(){}

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
    void operator=(const CSRPCG&) = delete;

};

}

