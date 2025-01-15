Info << "Enter EEqn" << endl;
assert(turbName == "laminar");
Info << "turbName : " << turbName << endl;
{
    volScalarField& he = thermo.he();
    start1 = MPI_Wtime();
#ifndef OPT_GenMatrix_E
    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
        (
            fvm::laplacian(turbulence->alpha(), he)
            - diffAlphaD
            + fvc::div(hDiffCorrFlux)
        )
    );
#else
    tmp<fvScalarMatrix> tEEqn = GenMatrix_E(rho, he, phi, K, dpdt, turbulence->alpha(), diffAlphaD, hDiffCorrFlux, linear_weights);
    fvScalarMatrix& EEqn = tEEqn.ref();
#endif
    end1 = MPI_Wtime();
    time_monitor_EEqn += double(end1 - start1);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1);

#ifdef OPT_GenMatrix_E_check
    fvScalarMatrix EEqn_answer
    (
        fvm::ddt(rho, he) 
        + fvm::div(phi, he)
        + fvc::ddt(rho, K) 
        + fvc::div(phi, K)
        - dpdt
        - fvm::laplacian(turbulence->alphaEff(), he)
        + diffAlphaD
         ==
            fvc::div(hDiffCorrFlux)
    );
    check_fvmatrix_equal(EEqn, EEqn_answer, "EEqn");
#endif

    // EEqn.relax();
    start1 = MPI_Wtime();
#ifdef USE_DF_MATRIX
    dfMatrix df_EEqn(EEqn);
    auto& psi = const_cast<volScalarField&>(EEqn.psi());
    auto& source = EEqn.source();
    auto& internalCoeffs = EEqn.internalCoeffs();
    auto& boundaryCoeffs = EEqn.boundaryCoeffs();
    df_EEqn.solve(psi, source, internalCoeffs, boundaryCoeffs, "ha");
#else
    EEqn.solve("ha");
#endif
    end1 = MPI_Wtime();

    time_monitor_EEqn += double(end1 - start1);
    time_monitor_EEqn_solve += double(end1 - start1);
}

Info << "Exit EEqn" << endl;