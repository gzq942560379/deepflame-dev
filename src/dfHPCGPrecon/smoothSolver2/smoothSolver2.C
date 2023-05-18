/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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

#include "smoothSolver2.H"
#include "symGaussSeidelSmoother.H"
// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(smoothSolver2, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<smoothSolver2>
        addsmoothSolver2SymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<smoothSolver2>
        addsmoothSolver2AsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::smoothSolver2::smoothSolver2
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{
    readControls();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::smoothSolver2::readControls()
{
    lduMatrix::solver::readControls();
    nSweeps_ = controlDict_.lookupOrDefault<label>("nSweeps", 4);
    //nSweeps_ = 4;
}


Foam::solverPerformance Foam::smoothSolver2::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // Setup class containing solver performance data
    solverPerformance solverPerf(typeName, fieldName_);

    // If the nSweeps_ is negative do a fixed number of sweeps
    // if (nSweeps_ < 0)
    // {
        autoPtr<lduMatrix::smoother> smootherPtr = lduMatrix::smoother::New
        (
            fieldName_,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            controlDict_
        );

        smootherPtr->smooth
        (
            psi,
            source,
            cmpt,
            nSweeps_
        );

        solverPerf.nIterations() -= nSweeps_;
    //}
    // else
    // {
    //     scalar normFactor = 0;

    //     {
    //         scalarField Apsi(psi.size());
    //         scalarField temp(psi.size());

    //         // Calculate A.psi
    //         matrix_.Amul(Apsi, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    //         // Calculate normalisation factor
    //         normFactor = this->normFactor(psi, source, Apsi, temp);

    //         // Calculate residual magnitude
    //         solverPerf.initialResidual() = gSumMag
    //         (
    //             (source - Apsi)(),
    //             matrix().mesh().comm()
    //         )/normFactor;
    //         solverPerf.finalResidual() = solverPerf.initialResidual();
    //     }

    //     if (lduMatrix::debug >= 2)
    //     {
    //         Info.masterStream(matrix().mesh().comm())
    //             << "   Normalisation factor = " << normFactor << endl;
    //     }


    //     // Check convergence, solve if not converged
    //     if
    //     (
    //         minIter_ > 0
    //      || !solverPerf.checkConvergence(tolerance_, relTol_)
    //     )
    //     {
    //         autoPtr<lduMatrix::smoother> smootherPtr = lduMatrix::smoother::New
    //         (
    //             fieldName_,
    //             matrix_,
    //             interfaceBouCoeffs_,
    //             interfaceIntCoeffs_,
    //             interfaces_,
    //             controlDict_
    //         );

    //         // Smoothing loop
    //         do
    //         {
    //             smootherPtr->smooth
    //             (
    //                 psi,
    //                 source,
    //                 cmpt,
    //                 nSweeps_
    //             );

    //             // Calculate the residual to check convergence
    //             solverPerf.finalResidual() = gSumMag
    //             (
    //                 matrix_.residual
    //                 (
    //                     psi,
    //                     source,
    //                     interfaceBouCoeffs_,
    //                     interfaces_,
    //                     cmpt
    //                 )(),
    //                 matrix().mesh().comm()
    //             )/normFactor;
    //         } while
    //         (
    //             (
    //                 (solverPerf.nIterations() += nSweeps_) < maxIter_
    //             && !solverPerf.checkConvergence(tolerance_, relTol_)
    //             )
    //          || solverPerf.nIterations() < minIter_
    //         );
    //     }
    // }

    return solverPerf;
}


// ************************************************************************* //
