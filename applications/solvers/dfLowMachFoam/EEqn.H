Info << "Enter EEqn" << endl;

{   
    clockTime EEqnClock;

    volScalarField& he = thermo.he();
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

    time_monitor_EEqn_pre += EEqnClock.timeIncrement();


#ifdef USE_DF_MATRIX
    dfMatrix df_EEqn(EEqn, regionPtr);
    auto& psi = const_cast<volScalarField&>(EEqn.psi());
    auto& source = EEqn.source();
    auto& internalCoeffs = EEqn.internalCoeffs();
    auto& boundaryCoeffs = EEqn.boundaryCoeffs();

    time_monitor_EEqn_convert += EEqnClock.timeIncrement();

    df_EEqn.solve(psi, source, internalCoeffs, boundaryCoeffs, "ha");
#else
    EEqn.solve("ha");
#endif

    time_monitor_EEqn_solve += EEqnClock.timeIncrement();

    time_monitor_EEqn += EEqnClock.elapsedTime();
}

Info << "Exit EEqn" << endl;