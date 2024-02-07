#include "DIVPCG.H"
#include <mpi.h>
#include "common_kernel.H"

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



// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

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
    df_triad(rAPtr, 1., sourcePtr, -1., wAPtr, 0., nCells);

    PCG_axpy_time += clock.timeIncrement();

    scalar gRASumMag = df_sum_mag(rAPtr, nCells);

    PCG_reduce_local_time += clock.timeIncrement();

    reduce(gRASumMag, sumOp<scalar>());

    PCG_allreduce_time += clock.timeIncrement();

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

            wArA = df_sum_prod(wAPtr, rAPtr, nCells);

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(wArA, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            if (solverPerf.nIterations() == 0)
            {
                df_copy(pAPtr, wAPtr, nCells);
                PCG_axpy_time += clock.timeIncrement();
            }
            else
            {
                scalar beta = wArA/wArAold;

                df_axpy(pAPtr, 1.0, wAPtr, beta, nCells);

                PCG_axpy_time += clock.timeIncrement();
            }

            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            
            PCG_spmv_time += clock.timeIncrement();

            scalar wApA = df_sum_prod(wAPtr, pAPtr, nCells);

            PCG_reduce_local_time += clock.timeIncrement();

            reduce(wApA, sumOp<scalar>());

            PCG_allreduce_time += clock.timeIncrement();

            // --- Update solution and residual:

            scalar alpha = wArA/wApA;

            df_axpy(psiPtr, alpha, pAPtr, 1., nCells);
            df_axpy(rAPtr, -alpha, wAPtr, 1., nCells);

            PCG_axpy_time += clock.timeIncrement();

            gRASumMag = df_sum_mag(rAPtr, nCells);

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


// ************************************************************************* //
