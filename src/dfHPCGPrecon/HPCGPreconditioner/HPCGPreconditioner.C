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

#include "HPCGPreconditioner.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(HPCGPreconditioner, 0);

    lduMatrix::preconditioner::addsymMatrixConstructorToTable
    <HPCGPreconditioner> addHPCGPreconditionerSymMatrixConstructorToTable_;

    lduMatrix::preconditioner::addasymMatrixConstructorToTable
    <HPCGPreconditioner> addHPCGPreconditionerAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::HPCGPreconditioner::HPCGPreconditioner
(
    const lduMatrix::solver& sol,
    const dictionary& solverControls
)
:
    HPCGSolver
    (
        sol.fieldName(),
        sol.matrix(),
        sol.interfaceBouCoeffs(),
        sol.interfaceIntCoeffs(),
        sol.interfaces(),
        solverControls
    ),
    lduMatrix::preconditioner(sol),
    nVcycles_(2)
{
    readControls();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::HPCGPreconditioner::~HPCGPreconditioner()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::HPCGPreconditioner::readControls()
{
    HPCGSolver::readControls();
    nVcycles_ = controlDict_.lookupOrDefault<label>("nVcycles", 2);
}


void Foam::HPCGPreconditioner::precondition
(
    scalarField& wA,
    const scalarField& rA,
    const direction cmpt
) const
{
    wA = 0.0;
    scalarField AwA(wA.size());
    scalarField finestCorrection(wA.size());
    scalarField finestResidual(rA);

    // Create coarse grid correction fields
    PtrList<scalarField> coarseCorrFields;

    // Create coarse grid sources
    PtrList<scalarField> coarseSources;

    // Create the smoothers for all levels
    PtrList<lduMatrix::smoother> smoothers;

    // Scratch fields if processor-agglomerated coarse level meshes
    // are bigger than original. Usually not needed
    scalarField ApsiScratch;
    scalarField finestCorrectionScratch;

    // Initialise the above data structures
    initVcycle
    (
        coarseCorrFields,
        coarseSources,
        smoothers,
        ApsiScratch,
        finestCorrectionScratch
    );

    for (label cycle=0; cycle<nVcycles_; cycle++)
    {
        Vcycle
        (
            smoothers,
            wA,
            rA,
            AwA,
            finestCorrection,
            finestResidual,

            (ApsiScratch.size() ? ApsiScratch : AwA),
            (
                finestCorrectionScratch.size()
              ? finestCorrectionScratch
              : finestCorrection
            ),

            coarseCorrFields,
            coarseSources,
            cmpt
        );

        if (cycle < nVcycles_-1)
        {
            matrix_.Amul(AwA, wA, interfaceBouCoeffs_, interfaces_, cmpt);
            finestResidual = rA;
            finestResidual -= AwA;
        }
    }
}


// ************************************************************************* //
