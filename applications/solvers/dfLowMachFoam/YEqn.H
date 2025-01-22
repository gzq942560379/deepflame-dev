Info << "Enter YEqn" << endl;

syncClockTime YEqnClock;

hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;

tmp<fv::convectionScheme<scalar>> mvConvection
(
    fv::convectionScheme<scalar>::New
    (
        mesh,
        fields,
        phi,
        mesh.divScheme("div(phi,Yi_h)")
    )
);

time_monitor_YEqn_pre += YEqnClock.timeIncrement();

// auto& mgcs = dynamic_cast<fv::multivariateGaussConvectionScheme<scalar>&>(mvConvection.ref());
// tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_ = mgcs.interpolationScheme()()(Y[0]);
// tmp<surfaceScalarField> tweights = tinterpScheme_().weights(Y[0]);
// const surfaceScalarField& weights = tweights();
// Info << "CPU weights\n" << weights << endl;
{
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
    // Info << "sumYDiffError\n" << sumYDiffError << endl;
    const surfaceScalarField phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();
    time_monitor_YEqn_pre += YEqnClock.timeIncrement();

    if (!splitting)
    {
        combustion->correct();
    }

    time_monitor_chem = YEqnClock.timeIncrement();

    volScalarField Yt(0.0*Y[0]);
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        
        time_monitor_YEqn_pre += YEqnClock.timeIncrement();

        if (i != inertIndex)
        {
            tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
            +
                (
                    turbName == "laminar"
                    ?  (mvConvection->fvmDiv(phi, Yi) + mvConvection->fvmDiv(phiUc, Yi))
                    :   mvConvection->fvmDiv(phi, Yi)
                )
            ==
                (
                    splitting
                    ?   fvm::laplacian(DEff(), Yi)
                    :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                    )
            );

            time_monitor_YEqn_pre += YEqnClock.timeIncrement();
            // YiEqn.relax();

#ifdef USE_DF_MATRIX
            dfMatrix df_YiEqn(YiEqn, regionPtr);
            auto& psi = const_cast<volScalarField&>(YiEqn.psi());
            auto& source = YiEqn.source();
            auto& internalCoeffs = YiEqn.internalCoeffs();
            auto& boundaryCoeffs = YiEqn.boundaryCoeffs();

            time_monitor_YEqn_convert += YEqnClock.timeIncrement();

            df_YiEqn.solve(psi, source, internalCoeffs, boundaryCoeffs, "Yi");
#else
            YiEqn.solve("Yi");
#endif

            time_monitor_YEqn_solve += YEqnClock.timeIncrement();

            Yi.max(0.0);
            Yt += Yi;

            time_monitor_YEqn_post += YEqnClock.timeIncrement();
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
    
}

time_monitor_YEqn_post += YEqnClock.timeIncrement(); 
time_monitor_YEqn += YEqnClock.elapsedTime();

time_monitor_YEqn -= time_monitor_chem;

Info << "Exit YEqn" << endl;