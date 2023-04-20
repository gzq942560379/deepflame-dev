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

#include "HPCGSolver.H"
//#include "PCG2.H"
#include "PBiCGStab.H"
#include "SubField.H"
#include"LUscalarMatrix.H"
#include "scalarMatrices.H"
#include "smoothSolver2.H"

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
Foam::solverPerformance Foam::HPCGSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    if(matrix_.symmetric())
    {
        Info << "The Matrix is symmetrix, so using PCG solver" << endl;
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalar wArA = solverPerf.great_;
    scalar wArAold = wArA;

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - wA);
    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);

    if (lduMatrix::debug >= 2)
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // --- Select and construct the preconditioner
        autoPtr<lduMatrix::preconditioner> preconPtr =
        lduMatrix::preconditioner::New
        (
            *this,
            controlDict_
        );

        // --- Solver iteration
        do
        {
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            preconPtr->precondition(wA, rA, cmpt);

            // --- Update search directions:
            wArA = gSumProd(wA, rA, matrix().mesh().comm());

            if (solverPerf.nIterations() == 0)
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
            }
            else
            {
                scalar beta = wArA/wArAold;

                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta*pAPtr[cell];
                }
            }


            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);

            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());


            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;


            // --- Update solution and residual:

            scalar alpha = wArA/wApA;

            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pAPtr[cell];
                rAPtr[cell] -= alpha*wAPtr[cell];
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;

        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    return solverPerf;
    }

    else
    {
    Info << "The Matrix is asymmetrix, so using PBiCGStab solver" << endl;
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - yA);
    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    const scalar normFactor = this->normFactor(psi, source, yA, pA);

    if (lduMatrix::debug >= 2)
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
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

        // --- Initial values not used
        scalar rA0rA = 0;
        scalar alpha = 0;
        scalar omega = 0;

        // --- Select and construct the preconditioner
        autoPtr<lduMatrix::preconditioner> preconPtr =
        lduMatrix::preconditioner::New
        (
            *this,
            controlDict_
        );

        // --- Solver iteration
        do
        {
            // --- Store previous rA0rA
            const scalar rA0rAold = rA0rA;

            rA0rA = gSumProd(rA0, rA, matrix().mesh().comm());

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(rA0rA)))
            {
                break;
            }

            // --- Update pA
            if (solverPerf.nIterations() == 0)
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = rAPtr[cell];
                }
            }
            else
            {
                // --- Test for singularity
                if (solverPerf.checkSingularity(mag(omega)))
                {
                    break;
                }

                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);

                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] =
                        rAPtr[cell] + beta*(pAPtr[cell] - omega*AyAPtr[cell]);
                }
            }

            // --- Precondition pA
            preconPtr->precondition(yA, pA, cmpt);

            // --- Calculate AyA
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);

            const scalar rA0AyA = gSumProd(rA0, AyA, matrix().mesh().comm());

            alpha = rA0rA/rA0AyA;

            // --- Calculate sA
            for (label cell=0; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha*AyAPtr[cell];
            }

            // --- Test sA for convergence
            solverPerf.finalResidual() =
                gSumMag(sA, matrix().mesh().comm())/normFactor;

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*yAPtr[cell];
                }

                solverPerf.nIterations()++;

                return solverPerf;
            }

            // --- Precondition sA
            preconPtr->precondition(zA, sA, cmpt);

            // --- Calculate tA
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);

            const scalar tAtA = gSumSqr(tA, matrix().mesh().comm());

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            omega = gSumProd(tA, sA, matrix().mesh().comm())/tAtA;

            // --- Update solution and residual
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*yAPtr[cell] + omega*zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega*tAPtr[cell];
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    return solverPerf;
    }
}

void Foam::HPCGSolver::Vcycle
(
    const PtrList<lduMatrix::smoother>& smoothers,
    scalarField& psi,
    const scalarField& source,
    scalarField& Apsi,
    scalarField& finestCorrection,
    scalarField& finestResidual,

    scalarField& scratch1,
    scalarField& scratch2,

    PtrList<scalarField>& coarseCorrFields,
    PtrList<scalarField>& coarseSources,
    const direction cmpt
) const
{
    const label coarsestLevel = matrixLevels_.size() - 1;
    agglomeration_.restrictField(coarseSources[0], finestResidual, 0, true);

    // Residual restriction (going to coarser levels)
    for (label leveli = 0; leveli < coarsestLevel; leveli++)
    //for (label leveli = 0; leveli <= coarsestLevel; leveli++)
    {
        if (coarseSources.set(leveli + 1))
        {
            // If the optional pre-smoothing sweeps are selected
            // smooth the coarse-grid field for the restricted source
            if (nPreSweeps_)
            {
                //Info << "=====level=====" << leveli << endl;
                //Info << "====num of cells====" << coarseSources[leveli].size() << endl;
                coarseCorrFields[leveli] = 0.0;

                smoothers[leveli + 1].smooth
                (
                    coarseCorrFields[leveli],
                    coarseSources[leveli],
                    cmpt,
                    min
                    (
                        nPreSweeps_ +  preSweepsLevelMultiplier_*leveli,
                        maxPreSweeps_
                    )
                );

                scalarField::subField ACf
                (
                    scratch1,
                    coarseCorrFields[leveli].size()
                );

                // Scale coarse-grid correction field
                // but not on the coarsest level because it evaluates to 1
                if (scaleCorrection_ && leveli < coarsestLevel - 1)
                {
                    scale
                    (
                        coarseCorrFields[leveli],
                        const_cast<scalarField&>
                        (
                            ACf.operator const scalarField&()
                        ),
                        matrixLevels_[leveli],
                        interfaceLevelsBouCoeffs_[leveli],
                        interfaceLevels_[leveli],
                        coarseSources[leveli],
                        cmpt
                    );
                }

                // Correct the residual with the new solution
                matrixLevels_[leveli].Amul
                (
                    const_cast<scalarField&>
                    (
                        ACf.operator const scalarField&()
                    ),
                    coarseCorrFields[leveli],
                    interfaceLevelsBouCoeffs_[leveli],
                    interfaceLevels_[leveli],
                    cmpt
                );

                coarseSources[leveli] -= ACf;
            }

            // Residual is equal to source
            agglomeration_.restrictField
            (
                coarseSources[leveli + 1],
                coarseSources[leveli],
                leveli + 1,
                true
            );
        }
    }


    if (coarseCorrFields.set(coarsestLevel) && solveCoarsestLevel_)
    {
        solveCoarsestLevel
        (
            coarseCorrFields[coarsestLevel],
            coarseSources[coarsestLevel]
        );
    }
    
    // Smooth the coareset level 
    if (coarseCorrFields.set(coarsestLevel) && smoothCoarsestLevel_)
    {
            
            Info << "smoothing the coareset level" << endl;
            smoothers[coarsestLevel + 1].smooth
            (
                coarseCorrFields[coarsestLevel],
                coarseSources[coarsestLevel],
                cmpt,
                nCoarsestSweeps_
            );
    }

    // Smoothing and prolongation of the coarse correction fields
    // (going to finer levels)

    scalarField dummyField(0);

    for (label leveli = coarsestLevel - 1; leveli >= 0; leveli--)
    {
        if (coarseCorrFields.set(leveli))
        {
            // Create a field for the pre-smoothed correction field
            // as a sub-field of the finestCorrection which is not
            // currently being used
            scalarField::subField preSmoothedCoarseCorrField
            (
                scratch2,
                coarseCorrFields[leveli].size()
            );

            // Only store the preSmoothedCoarseCorrField if pre-smoothing is
            // used
            if (nPreSweeps_)
            {
                preSmoothedCoarseCorrField = coarseCorrFields[leveli];
            }

            agglomeration_.prolongField
            (
                coarseCorrFields[leveli],
                (
                    coarseCorrFields.set(leveli + 1)
                  ? coarseCorrFields[leveli + 1]
                  : dummyField              // dummy value
                ),
                leveli + 1,
                true
            );


            // Create A.psi for this coarse level as a sub-field of Apsi
            scalarField::subField ACf
            (
               scratch1,
                coarseCorrFields[leveli].size()
            );
            scalarField& ACfRef =
                const_cast<scalarField&>(ACf.operator const scalarField&());

            if (interpolateCorrection_) // && leveli < coarsestLevel - 2)
            {
                if (coarseCorrFields.set(leveli+1))
                {
                    interpolate
                    (
                        coarseCorrFields[leveli],
                        ACfRef,
                        matrixLevels_[leveli],
                        interfaceLevelsBouCoeffs_[leveli],
                        interfaceLevels_[leveli],
                        agglomeration_.restrictAddressing(leveli + 1),
                        coarseCorrFields[leveli + 1],
                        cmpt
                    );
                }
                else
                {
                    interpolate
                    (
                        coarseCorrFields[leveli],
                        ACfRef,
                        matrixLevels_[leveli],
                        interfaceLevelsBouCoeffs_[leveli],
                        interfaceLevels_[leveli],
                        cmpt
                    );
                }
            }

            // Scale coarse-grid correction field
            // but not on the coarsest level because it evaluates to 1
            if
            (
                scaleCorrection_
             && (interpolateCorrection_ || leveli < coarsestLevel - 1)
            )
            {
                scale
                (
                    coarseCorrFields[leveli],
                    ACfRef,
                    matrixLevels_[leveli],
                    interfaceLevelsBouCoeffs_[leveli],
                    interfaceLevels_[leveli],
                    coarseSources[leveli],
                    cmpt
                );
            }

            // Only add the preSmoothedCoarseCorrField if pre-smoothing is
            // used
            if (nPreSweeps_)
            {
                coarseCorrFields[leveli] += preSmoothedCoarseCorrField;
            }

            smoothers[leveli + 1].smooth
            (
                coarseCorrFields[leveli],
                coarseSources[leveli],
                cmpt,
                min
                (
                    nPostSweeps_ + postSweepsLevelMultiplier_*leveli,
                    maxPostSweeps_
                )
            );
        }
    }

    // Prolong the finest level correction
    agglomeration_.prolongField
    (
        finestCorrection,
        coarseCorrFields[0],
        0,
        true
    );

    if (interpolateCorrection_)
    {
        interpolate
        (
            finestCorrection,
            Apsi,
            matrix_,
            interfaceBouCoeffs_,
            interfaces_,
            agglomeration_.restrictAddressing(0),
            coarseCorrFields[0],
            cmpt
        );
    }

    if (scaleCorrection_)
    {
        // Scale the finest level correction
        scale
        (
            finestCorrection,
            Apsi,
            matrix_,
            interfaceBouCoeffs_,
            interfaces_,
            finestResidual,
            cmpt
        );
    }

    forAll(psi, i)
    {
        psi[i] += finestCorrection[i];
    }

    //Info << "===before finest smooothing===" << psi << endl;
    //Info << "===size of psi===" << psi.size() << endl;
    smoothers[0].smooth
    (
        psi,
        source,
        cmpt,
        nFinestSweeps_
    );

    // Info << "===after finest smooothing===" << psi << endl;
}


void Foam::HPCGSolver::initVcycle
(
    PtrList<scalarField>& coarseCorrFields,
    PtrList<scalarField>& coarseSources,
    PtrList<lduMatrix::smoother>& smoothers,
    scalarField& scratch1,
    scalarField& scratch2
) const
{
    label maxSize = matrix_.diag().size();

    coarseCorrFields.setSize(matrixLevels_.size());
    coarseSources.setSize(matrixLevels_.size());
    smoothers.setSize(matrixLevels_.size() + 1);
    //Info << "The size of the smoother is" << matrixLevels_.size() + 1 << endl;

    // Create the smoother for the finest level
    smoothers.set
    (
        0,
        lduMatrix::smoother::New
        (
            fieldName_,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            controlDict_
        )
    );

    forAll(matrixLevels_, leveli)
    {
        if (agglomeration_.nCells(leveli) >= 0)
        {
            label nCoarseCells = agglomeration_.nCells(leveli);

            coarseSources.set(leveli, new scalarField(nCoarseCells));
        }

        if (matrixLevels_.set(leveli))
        {
            const lduMatrix& mat = matrixLevels_[leveli];

            label nCoarseCells = mat.diag().size();

            maxSize = max(maxSize, nCoarseCells);

            coarseCorrFields.set(leveli, new scalarField(nCoarseCells));

            smoothers.set
            (
                leveli + 1,
                lduMatrix::smoother::New
                (
                    fieldName_,
                    matrixLevels_[leveli],
                    interfaceLevelsBouCoeffs_[leveli],
                    interfaceLevelsIntCoeffs_[leveli],
                    interfaceLevels_[leveli],
                    controlDict_
                )
            );
        }
    }

    if (maxSize > matrix_.diag().size())
    {
        // Allocate some scratch storage
        scratch1.setSize(maxSize);
        scratch2.setSize(maxSize);
    }
}


// Foam::dictionary Foam::HPCGSolver::PCG2solverDict
// (
//     const scalar tol,
//     const scalar relTol
// ) const
// {
//     dictionary dict(IStringStream("solver PCG2; preconditioner DIC;")());
//     dict.add("tolerance", tol);
//     dict.add("relTol", relTol);

//     return dict;
// }


// Foam::dictionary Foam::HPCGSolver::PBiCGStabSolverDict
// (
//     const scalar tol,
//     const scalar relTol
// ) const
// {
//     dictionary dict(IStringStream("solver PBiCGStab; preconditioner DILU;")());
//     dict.add("tolerance", tol);
//     dict.add("relTol", relTol);

//     return dict;
// }

Foam::dictionary Foam::HPCGSolver::smoothSolver2Dict
(
) const
{
    dictionary dict(IStringStream("solver smoothSolver; smoother symGaussSeidel;")());

    return dict;
}


void Foam::HPCGSolver::solveCoarsestLevel
(
    scalarField& coarsestCorrField,
    const scalarField& coarsestSource
) const
{
    const label coarsestLevel = matrixLevels_.size() - 1;

    label coarseComm = matrixLevels_[coarsestLevel].mesh().comm();
    //convert(matrixLevels_[coarsestLevel],interfaceLevelsBouCoeffs_[coarsestLevel],interfaceLevels_[coarsestLevel]);
    if (directSolveCoarsest_)
    {
        //label nCells = coarsestSource.size();
        //Info << "====before solving======" << coarsestCorrField << endl;
        coarsestLUMatrixPtr_->solve(coarsestCorrField, coarsestSource);
        //Info << "===after solving=======" << coarsestCorrField << endl;
    }
    // else if
    //(
    //    agglomeration_.processorAgglomerate()
    // && procMatrixLevels_.set(coarsestLevel)
    //)
    //{
    //    // const labelList& agglomProcIDs = agglomeration_.agglomProcIDs
    //    //(
    //    //    coarsestLevel
    //    //);
    //    //
    //    // scalarField allSource;
    //    //
    //    // globalIndex cellOffsets;
    //    // if (Pstream::myProcNo(coarseComm) == agglomProcIDs[0])
    //    //{
    //    //    cellOffsets.offsets() =
    //    //        agglomeration_.cellOffsets(coarsestLevel);
    //    //}
    //    //
    //    // cellOffsets.gather
    //    //(
    //    //    coarseComm,
    //    //    agglomProcIDs,
    //    //    coarsestSource,
    //    //    allSource
    //    //);
    //    //
    //    // scalarField allCorrField;
    //    // solverPerformance coarseSolverPerf;
    //
    //    label solveComm = agglomeration_.procCommunicator(coarsestLevel);
    //
    //    coarsestCorrField = 0;
    //    solverPerformance coarseSolverPerf;
    //
    //    if (Pstream::myProcNo(solveComm) != -1)
    //    {
    //        const lduMatrix& allMatrix = procMatrixLevels_[coarsestLevel];
    //
    //        {
    //            Pout<< "** Master:Solving on comm:" << solveComm
    //                << " with procs:" << UPstream::procID(solveComm) << endl;
    //
    //            if (allMatrix.asymmetric())
    //            {
    //                coarseSolverPerf = PBiCGStab
    //                (
    //                    "coarsestLevelCorr",
    //                    allMatrix,
    //                    procInterfaceLevelsBouCoeffs_[coarsestLevel],
    //                    procInterfaceLevelsIntCoeffs_[coarsestLevel],
    //                    procInterfaceLevels_[coarsestLevel],
    //                    PBiCGStabSolverDict(tolerance_, relTol_)
    //                ).solve
    //                (
    //                    coarsestCorrField,
    //                    coarsestSource
    //                );
    //            }
    //            else
    //            {
    //                coarseSolverPerf = PCG
    //                (
    //                    "coarsestLevelCorr",
    //                    allMatrix,
    //                    procInterfaceLevelsBouCoeffs_[coarsestLevel],
    //                    procInterfaceLevelsIntCoeffs_[coarsestLevel],
    //                    procInterfaceLevels_[coarsestLevel],
    //                    PCGsolverDict(tolerance_, relTol_)
    //                ).solve
    //                (
    //                    coarsestCorrField,
    //                    coarsestSource
    //                );
    //            }
    //        }
    //    }
    //
    //    Pout<< "done master solve." << endl;
    //
    //    //// Scatter to all processors
    //    // coarsestCorrField.setSize(coarsestSource.size());
    //    // cellOffsets.scatter
    //    //(
    //    //    coarseComm,
    //    //    agglomProcIDs,
    //    //    allCorrField,
    //    //    coarsestCorrField
    //    //);
    //
    //    if (debug >= 2)
    //    {
    //        coarseSolverPerf.print(Info.masterStream(coarseComm));
    //    }
    //
    //    Pout<< "procAgglom: coarsestSource   :" << coarsestSource << endl;
    //    Pout<< "procAgglom: coarsestCorrField:" << coarsestCorrField << endl;
    //}
    else
    {
        coarsestCorrField = 0;
        solverPerformance coarseSolverPerf;

        if(smoothsolver_)
        {
            coarseSolverPerf = smoothSolver2
            (
                "coarsestLevelCorr",
                matrixLevels_[coarsestLevel],
                interfaceLevelsBouCoeffs_[coarsestLevel],
                interfaceLevelsIntCoeffs_[coarsestLevel],
                interfaceLevels_[coarsestLevel],
                smoothSolver2Dict()
            ).solve
            (
                coarsestCorrField,
                coarsestSource
            );            

        }
        
        
        // else if (matrixLevels_[coarsestLevel].asymmetric())
        // {
        //     coarseSolverPerf = PBiCGStab
        //     (
        //         "coarsestLevelCorr",
        //         matrixLevels_[coarsestLevel],
        //         interfaceLevelsBouCoeffs_[coarsestLevel],
        //         interfaceLevelsIntCoeffs_[coarsestLevel],
        //         interfaceLevels_[coarsestLevel],
        //         PBiCGStabSolverDict(tolerance_, relTol_)
        //     ).solve
        //     (
        //         coarsestCorrField,
        //         coarsestSource
        //     );
        // }
        // else
        // {
        //     coarseSolverPerf = PCG2
        //     (
        //         "coarsestLevelCorr",
        //         matrixLevels_[coarsestLevel],
        //         interfaceLevelsBouCoeffs_[coarsestLevel],
        //         interfaceLevelsIntCoeffs_[coarsestLevel],
        //         interfaceLevels_[coarsestLevel],
        //         PCG2solverDict(tolerance_, relTol_)
        //     ).solve
        //     (
        //         coarsestCorrField,
        //         coarsestSource
        //     );
        // }

        //if (debug >= 2)
        //{
            coarseSolverPerf.print(Info.masterStream(coarseComm));
        //}
    }
}


// ************************************************************************* //
