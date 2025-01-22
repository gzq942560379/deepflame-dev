Info << "Enter EEqn" << endl;

{
    volScalarField& he = thermo.he();
    start1 = MPI_Wtime();
    fvScalarMatrix EEqn
    (

        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
        (
            turbName == "laminar"
            ?
            (
                fvm::laplacian(turbulence->alpha(), he)
            -   diffAlphaD
            +   fvc::div(hDiffCorrFlux)
            )
            :
            (
                fvm::laplacian(turbulence->alphaEff(), he)
            )
        )
    );
    end1 = MPI_Wtime();
    time_monitor_EEqn += double(end1 - start1);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1);

    // EEqn.relax();
    start1 = MPI_Wtime();

#ifdef USE_DF_MATRIX
    dfMatrix df_EEqn(EEqn, regionPtr);
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