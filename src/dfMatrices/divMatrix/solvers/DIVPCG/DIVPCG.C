#include "DIVPCG.H"
#include <mpi.h>

namespace Foam
{
    defineTypeNameAndDebug(DIVPCG, 0);

    divMatrix::solver::addsymMatrixConstructorToTable<DIVPCG>
        addDIVPCGSymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DIVPCG::DIVPCG
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

// #define DIVPCG_solve_naive
#define DIVPCG_solve_no_norm

#if defined(DIVPCG_solve_naive)


Foam::solverPerformance Foam::DIVPCG::solve
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

    label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalar wArA = solverPerf.great_;
    scalar wArAold = wArA;

    PCG_misc_time += clock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    PCG_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
    scalarField rA(source - wA);

    PCG_axpy_time += clock.timeIncrement();

    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);

    PCG_norm_time += clock.timeIncrement();

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
    
    PCG_reduce_time += clock.timeIncrement();

    solverPerf.finalResidual() = solverPerf.initialResidual();
    // --- Check convergence, solve if not converged

    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // --- Solver iteration
        do
        {
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            preconPtr_->precondition(wA, rA, cmpt);
            PCG_precondition_time += clock.timeIncrement();

            // --- Update search directions:
            wArA = gSumProd(wA, rA, matrix().mesh().comm());
            PCG_reduce_time += clock.timeIncrement();


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
                    pAPtr[cell] = wAPtr[cell] + beta * pAPtr[cell];
                }
            }
            PCG_axpy_time += clock.timeIncrement();

            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            
            PCG_spmv_time += clock.timeIncrement();

            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());

            PCG_reduce_time += clock.timeIncrement();

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;
            // --- Update solution and residual:

            scalar alpha = wArA/wApA;
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha * pAPtr[cell];
                rAPtr[cell]  -= alpha * wAPtr[cell];
            }
            PCG_axpy_time += clock.timeIncrement();

            solverPerf.finalResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
            
            PCG_reduce_time += clock.timeIncrement();
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );

    }
    PCG_time += clock.elapsedTime();

    return solverPerf;
}


#elif defined(DIVPCG_solve_omp)


Foam::solverPerformance Foam::DIVPCG::solve
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

    label nCells = psi.size();

    label gNCells = nCells;
    reduce(gNCells, sumOp<label>());
    PCG_allreduce_time += clock.timeIncrement();

    const scalar* __restrict__ sourcePtr = source.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalarField rA(nCells);
    scalar* __restrict__ rAPtr = rA.begin();

    scalar wArA = solverPerf.great_;
    scalar wArAold = wArA;

    PCG_misc_time += clock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    PCG_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label c = 0; c < nCells; ++c){
        rAPtr[c] = sourcePtr[c] - wAPtr[c];
    }

    PCG_axpy_time += clock.timeIncrement();
    
    matrix_.sumA(pA, interfaceBouCoeffs_, interfaces_);

    PCG_sumA_time += clock.timeIncrement();

    scalar gPsiSum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gPsiSum)
#endif
    for(label c = 0; c < nCells; ++c){
        gPsiSum += psi[c];
    }

    PCG_reduce_local_time += clock.timeIncrement();

    reduce(gPsiSum, sumOp<scalar>());

    PCG_allreduce_time += clock.timeIncrement();

    scalar gPsiAvg = gPsiSum / gNCells;

    scalar gTmpSum = 0.;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:gTmpSum)
#endif
    for(label c = 0; c < nCells; ++c){
        double tmp = pAPtr[c] * gPsiAvg;
        gTmpSum += std::abs(wAPtr[c] - tmp) + std::abs(sourcePtr[c] - tmp);
    }


    reduce(gTmpSum, sumOp<scalar>());

    PCG_norm_time += clock.timeIncrement();
    
    scalar normFactor = gTmpSum + solverPerformance::small_;

    scalar gRASumMag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gRASumMag)
#endif
    for(label c = 0; c < nCells; ++c){
        gRASumMag += std::abs(rAPtr[c]);
    }

    PCG_reduce_local_time += clock.timeIncrement();

    reduce(gRASumMag, sumOp<scalar>());

    PCG_allreduce_time += clock.timeIncrement();

    solverPerf.initialResidual() = gRASumMag / normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();
    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // --- Solver iteration
        do
        {
            PCG_misc_time += clock.timeIncrement();
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            preconPtr_->precondition(wA, rA, cmpt);

            PCG_precondition_time += clock.timeIncrement();

            wArA = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:wArA)
#endif
            for(label c = 0; c < nCells; ++c){
                wArA += wAPtr[c] * rAPtr[c];
            }

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(wArA, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            if (solverPerf.nIterations() == 0)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
                PCG_axpy_time += clock.timeIncrement();
            }
            else
            {
                scalar beta = wArA/wArAold;
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta * pAPtr[cell];
                }

                PCG_axpy_time += clock.timeIncrement();
            }

            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            
            PCG_spmv_time += clock.timeIncrement();

            scalar wApA = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:wApA)
#endif
            for(label c = 0; c < nCells; ++c){
                wApA += wAPtr[c] * pAPtr[c];
            }

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(wApA, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            // --- Update solution and residual:

            scalar alpha = wArA/wApA;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:wArA)
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha * pAPtr[cell];
                rAPtr[cell]  -= alpha * wAPtr[cell];
            }

            PCG_axpy_time += clock.timeIncrement();

            gRASumMag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gRASumMag)
#endif
            for(label c = 0; c < nCells; ++c){
                gRASumMag += std::abs(rAPtr[c]);
            }

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(gRASumMag, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            solverPerf.finalResidual() = gRASumMag / normFactor;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );

    }
    PCG_time += clock.elapsedTime();

    return solverPerf;
}

#elif defined(DIVPCG_solve_no_norm)

Foam::solverPerformance Foam::DIVPCG::solve
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

    label nCells = psi.size();

    const scalar* __restrict__ sourcePtr = source.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalarField rA(nCells);
    scalar* __restrict__ rAPtr = rA.begin();

    scalar wArA = solverPerf.great_;
    scalar wArAold = wArA;

    PCG_misc_time += clock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    PCG_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label c = 0; c < nCells; ++c){
        rAPtr[c] = sourcePtr[c] - wAPtr[c];
    }

    PCG_axpy_time += clock.timeIncrement();

    scalar gRASumMag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gRASumMag)
#endif
    for(label c = 0; c < nCells; ++c){
        gRASumMag += std::abs(rAPtr[c]);
    }

    PCG_reduce_local_time += clock.timeIncrement();

    reduce(gRASumMag, sumOp<scalar>());

    PCG_allreduce_time += clock.timeIncrement();

    // solverPerf.initialResidual() = gRASumMag / normFactor;
    solverPerf.initialResidual() = gRASumMag;
    solverPerf.finalResidual() = solverPerf.initialResidual();
    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // --- Solver iteration
        do
        {
            PCG_misc_time += clock.timeIncrement();
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            preconPtr_->precondition(wA, rA, cmpt);

            PCG_precondition_time += clock.timeIncrement();

            wArA = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:wArA)
#endif
            for(label c = 0; c < nCells; ++c){
                wArA += wAPtr[c] * rAPtr[c];
            }

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(wArA, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            if (solverPerf.nIterations() == 0)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
                PCG_axpy_time += clock.timeIncrement();
            }
            else
            {
                scalar beta = wArA/wArAold;
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta * pAPtr[cell];
                }

                PCG_axpy_time += clock.timeIncrement();
            }

            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            
            PCG_spmv_time += clock.timeIncrement();

            scalar wApA = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:wApA)
#endif
            for(label c = 0; c < nCells; ++c){
                wApA += wAPtr[c] * pAPtr[c];
            }

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(wApA, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            // --- Update solution and residual:

            scalar alpha = wArA/wApA;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:wArA)
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha * pAPtr[cell];
                rAPtr[cell]  -= alpha * wAPtr[cell];
            }

            PCG_axpy_time += clock.timeIncrement();

            gRASumMag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gRASumMag)
#endif
            for(label c = 0; c < nCells; ++c){
                gRASumMag += std::abs(rAPtr[c]);
            }

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(gRASumMag, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            solverPerf.finalResidual() = gRASumMag;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );

    }
    PCG_time += clock.elapsedTime();

    matrix_.flops_ += (6 * solverPerf.nIterations() + 2) * 2 * nCells;
    return solverPerf;
}

#else

#error "DIVPCG_solve"

#endif

// ************************************************************************* //
