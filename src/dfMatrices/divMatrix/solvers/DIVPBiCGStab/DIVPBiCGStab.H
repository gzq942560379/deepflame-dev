/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2019 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::DIVPBiCGStab

Description
    Preconditioned bi-conjugate gradient stabilized solver for asymmetric
    lduMatrices using a run-time selectable preconditioner.

    References:
    \verbatim
        Van der Vorst, H. A. (1992).
        Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG
        for the solution of nonsymmetric linear systems.
        SIAM Journal on scientific and Statistical Computing, 13(2), 631-644.

        Barrett, R., Berry, M. W., Chan, T. F., Demmel, J., Donato, J.,
        Dongarra, J., Eijkhout, V., Pozo, R., Romine, C. & Van der Vorst, H.
        (1994).
        Templates for the solution of linear systems:
        building blocks for iterative methods
        (Vol. 43). Siam.
    \endverbatim

SourceFiles
    DIVPBiCGStab.C

\*---------------------------------------------------------------------------*/

#ifndef DIVPBiCGStab_H
#define DIVPBiCGStab_H

#include "divMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class DIVPBiCGStab Declaration
\*---------------------------------------------------------------------------*/

class DIVPBiCGStab
:
    public divMatrix::solver
{
    autoPtr<divMatrix::preconditioner> preconPtr_;

public:

    //- Runtime type information
    TypeName("PBiCGStab");


    // Constructors

        //- Construct from matrix components and solver data stream
        DIVPBiCGStab
        (
            const word& fieldName,
            const divMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        DIVPBiCGStab(const DIVPBiCGStab&) = delete;


    //- Destructor
    virtual ~DIVPBiCGStab()
    {}


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
        void operator=(const DIVPBiCGStab&) = delete;

    // profiling
    mutable double PBiCGStab_time = 0.;
    mutable double PBiCGStab_spmv_time = 0.;
    mutable double PBiCGStab_normFactor_time = 0.;
    mutable double PBiCGStab_gSumMag_time = 0.;
    mutable double PBiCGStab_gSumProd_time = 0.;
    mutable double PBiCGStab_gSumSqr_time = 0.;
    mutable double PBiCGStab_localUpdate_time = 0.;
    mutable double PBiCGStab_precondition_time = 0.;
    mutable double PBiCGStab_allreduce_time = 0.;
    mutable double PBiCGStab_misc_time = 0.;

    void print_time() const {
        double PBiCGStab_other_time = PBiCGStab_time - PBiCGStab_spmv_time - PBiCGStab_normFactor_time - PBiCGStab_gSumMag_time \
            - PBiCGStab_gSumProd_time - PBiCGStab_gSumSqr_time - PBiCGStab_localUpdate_time - PBiCGStab_precondition_time \
            - PBiCGStab_misc_time;
        Info << "DIVPBiCGStab solve profiling ------------------------------------------------------------" << endl;
        Info << "PBiCGStab time : " << PBiCGStab_time << endl;
        Info << "PBiCGStab spmv time : " << PBiCGStab_spmv_time << ", " << PBiCGStab_spmv_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab normFactor time : " << PBiCGStab_normFactor_time << ", " << PBiCGStab_normFactor_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab gSumMag time : " << PBiCGStab_gSumMag_time << ", " << PBiCGStab_gSumMag_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab gSumProd time : " << PBiCGStab_gSumProd_time << ", " << PBiCGStab_gSumProd_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab gSumSqr time : " << PBiCGStab_gSumSqr_time << ", " << PBiCGStab_gSumSqr_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab localUpdate time : " << PBiCGStab_localUpdate_time << ", " << PBiCGStab_localUpdate_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab precondition time : " << PBiCGStab_precondition_time << ", " << PBiCGStab_precondition_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab misc time : " << PBiCGStab_misc_time << ", " << PBiCGStab_misc_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "PBiCGStab other time : " << PBiCGStab_other_time << ", " << PBiCGStab_other_time / PBiCGStab_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
