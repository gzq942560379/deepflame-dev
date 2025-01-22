Info << "Enter UEqn" << endl;
assert(pimple.momentumPredictor());

clockTime UEqnClock;

// Solve the Momentum equation
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
+ turbulence->divDevRhoReff(U) 
);
fvVectorMatrix& UEqn = tUEqn.ref();

time_monitor_UEqn_pre += UEqnClock.timeIncrement();

{
    auto tUEqnSolve = (UEqn == -fvc::grad(p));
    auto& UEqnSolve = tUEqnSolve.ref();

    time_monitor_UEqn_pre += UEqnClock.timeIncrement();

#ifdef USE_DF_MATRIX
    dfMatrix df_UEqnSolve(tUEqnSolve.ref(), regionPtr);
    auto& psi = const_cast<volVectorField&>(UEqnSolve.psi());
    auto& source = UEqnSolve.source();
    auto& internalCoeffs = UEqnSolve.internalCoeffs();
    auto& boundaryCoeffs = UEqnSolve.boundaryCoeffs();

    time_monitor_UEqn_convert += UEqnClock.timeIncrement();

    df_UEqnSolve.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
    UEqnSolve.solve();
#endif

    time_monitor_UEqn_solve += UEqnClock.timeIncrement(); 

    K.oldTime();
    K = 0.5*magSqr(U);

}

time_monitor_UEqn_post += UEqnClock.timeIncrement();

time_monitor_UEqn += UEqnClock.elapsedTime();

Info << "Exit UEqn" << endl;