/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2018 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "DIVPBiCGStab.H"
#include <mpi.h>
#include <clockTime.H>
#include "common_kernel.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(DIVPBiCGStab, 0);

    divMatrix::solver::addsymMatrixConstructorToTable<DIVPBiCGStab>
        addDIVPBiCGStabSymMatrixConstructorToTable_;

    divMatrix::solver::addasymMatrixConstructorToTable<DIVPBiCGStab>
        addDIVPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DIVPBiCGStab::DIVPBiCGStab
(
    const word& fieldName,
    const divMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    divMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    preconPtr_(
        divMatrix::preconditioner::New
        (
            *this,
            controlDict_
        )
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::DIVPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    clockTime clock;

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        divMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    label gNCells = nCells;
    reduce(gNCells, sumOp<label>());

    const scalar* __restrict__ sourcePtr = source.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    scalarField rA(nCells);
    scalar* __restrict__ rAPtr = rA.begin();


    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    
    PBiCGStab_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
    df_triad(rAPtr, 1., sourcePtr, -1., yAPtr, 0., nCells);

    PBiCGStab_localUpdate_time += clock.timeIncrement();

    // --- Calculate normalisation factor
    matrix_.sumA(pA, interfaceBouCoeffs_, interfaces_);

    scalar gPsiSum = df_sum(psiPtr, nCells);

    reduce(gPsiSum, sumOp<scalar>());

    scalar gPsiAvg = gPsiSum / gNCells;

    // norm factor local
    scalar gTmpSum = df_norm_factor_local(pAPtr, gPsiAvg, yAPtr, sourcePtr, nCells);

    reduce(gTmpSum, sumOp<scalar>());
    
    scalar normFactor = gTmpSum + solverPerformance::small_;

    PBiCGStab_normFactor_time += clock.timeIncrement();

    // --- Calculate normalised residual norm
    scalar gRASumMag = df_sum_mag(rAPtr, nCells);

    reduce(gRASumMag, sumOp<scalar>());
    
    solverPerf.initialResidual() = gRASumMag / normFactor;

    PBiCGStab_gSumMag_time += clock.timeIncrement();

    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        PBiCGStab_misc_time += clock.timeIncrement();

        scalarField AyA(nCells);
        scalar* __restrict__ AyAPtr = AyA.begin();

        scalarField sA(nCells);
        scalar* __restrict__ sAPtr = sA.begin();

        scalarField zA(nCells);
        scalar* __restrict__ zAPtr = zA.begin();

        scalarField tA(nCells);
        scalar* __restrict__ tAPtr = tA.begin();

        // --- Store initial residual
        const scalarField rA0(rA);
        const scalar* __restrict__ rA0Ptr = rA0.begin();

        // --- Initial values not used
        scalar rA0rA = 0;
        scalar alpha = 0;
        scalar omega = 0;

        // --- Solver iteration
        do
        {
            // --- Store previous rA0rA
            const scalar rA0rAold = rA0rA;

            PBiCGStab_misc_time += clock.timeIncrement();

            rA0rA = df_sum_prod(rA0Ptr, rAPtr, nCells);

            reduce(rA0rA, sumOp<scalar>());

            PBiCGStab_gSumProd_time += clock.timeIncrement();

            // --- Update pA
            if (solverPerf.nIterations() == 0)
            {
                df_copy(pAPtr, rAPtr, nCells);

                PBiCGStab_localUpdate_time += clock.timeIncrement();
            }
            else
            {
                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);

                df_triad(pAPtr, 1.0, rAPtr, - beta * omega, AyAPtr, beta, nCells);

                PBiCGStab_localUpdate_time += clock.timeIncrement();
            }

            // --- Precondition pA
            preconPtr_->precondition(yA, pA, cmpt);
            PBiCGStab_precondition_time += clock.timeIncrement();
            
            // --- Calculate AyA
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);
            PBiCGStab_spmv_time += clock.timeIncrement();


            scalar rA0AyA = df_sum_prod(rA0Ptr, AyAPtr, nCells);

            reduce(rA0AyA, sumOp<scalar>());

            PBiCGStab_gSumProd_time += clock.timeIncrement();

            // --- Calculate sA
            alpha = rA0rA/rA0AyA;
            // --- Calculate sA
            df_triad(sAPtr, 1.0, rAPtr, -alpha, AyAPtr, 0., nCells);

            PBiCGStab_localUpdate_time += clock.timeIncrement();

            // --- Test sA for convergence
            scalar gSASumMag = df_sum_mag(sAPtr, nCells);

            reduce(gSASumMag, sumOp<scalar>());

            solverPerf.finalResidual() = gSASumMag / normFactor;

            PBiCGStab_gSumMag_time += clock.timeIncrement();

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                PBiCGStab_misc_time += clock.timeIncrement();

                df_axpy(psiPtr, alpha, yAPtr, 1., nCells);

                solverPerf.nIterations()++;
                PBiCGStab_localUpdate_time += clock.timeIncrement();
                break;
            }

            // --- Precondition sA
            preconPtr_->precondition(zA, sA, cmpt);
            
            PBiCGStab_precondition_time += clock.timeIncrement();

            // --- Calculate tA
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);
            PBiCGStab_spmv_time += clock.timeIncrement();

            scalar tAtA = df_sum_sqr(tAPtr, nCells);

            reduce(tAtA, sumOp<scalar>());
            
            PBiCGStab_gSumSqr_time += clock.timeIncrement();

            omega = df_sum_prod(tAPtr, sAPtr, nCells);

            reduce(omega, sumOp<scalar>());

            omega /= tAtA;
            
            PBiCGStab_gSumProd_time += clock.timeIncrement();

            // --- Update solution and residual
            df_triad(psiPtr, alpha, yAPtr, omega, zAPtr, 1., nCells);
            df_triad(rAPtr, 1., sAPtr, - omega, tAPtr, 0., nCells);

            PBiCGStab_localUpdate_time += clock.timeIncrement();

            gRASumMag = df_sum_mag(rAPtr, nCells);

            reduce(gRASumMag, sumOp<scalar>());
            
            solverPerf.finalResidual() = gRASumMag / normFactor;

            PBiCGStab_gSumMag_time += clock.timeIncrement();
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    PBiCGStab_time += clock.elapsedTime();
    return solverPerf;
}


// ************************************************************************* //
