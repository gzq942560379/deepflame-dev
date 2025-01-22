Info << "Enter YEqn" << endl;

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

// auto& mgcs = dynamic_cast<fv::multivariateGaussConvectionScheme<scalar>&>(mvConvection.ref());
// tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_ = mgcs.interpolationScheme()()(Y[0]);
// tmp<surfaceScalarField> tweights = tinterpScheme_().weights(Y[0]);
// const surfaceScalarField& weights = tweights();
// Info << "CPU weights\n" << weights << endl;
{
    start1 = MPI_Wtime();
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
    // Info << "sumYDiffError\n" << sumYDiffError << endl;
    const surfaceScalarField phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();
    start1 = MPI_Wtime();
    time_monitor_YEqn += double(end1 - start1);
    time_monitor_YEqn_solve += double(end1 - start1);  

    //MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

    if (!splitting)
    {
        start1 = MPI_Wtime();
        combustion->correct();
        end1 = MPI_Wtime();
        //int flag_mpi_init;
        //MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        time_monitor_chem += end1 - start1;
    }

    start2 = MPI_Wtime();
    volScalarField Yt(0.0*Y[0]);
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        if (i != inertIndex)
        {
            start1 = MPI_Wtime();
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
            
            end1 = MPI_Wtime();
            time_monitor_YEqn_mtxAssembly += double(end1 - start1);
            // YiEqn.relax();

            start1 = MPI_Wtime();
#ifdef USE_DF_MATRIX
            dfMatrix df_YiEqn(YiEqn, regionPtr);
            auto& psi = const_cast<volScalarField&>(YiEqn.psi());
            auto& source = YiEqn.source();
            auto& internalCoeffs = YiEqn.internalCoeffs();
            auto& boundaryCoeffs = YiEqn.boundaryCoeffs();
            df_YiEqn.solve(psi, source, internalCoeffs, boundaryCoeffs, "Yi");
#else
            YiEqn.solve("Yi");
#endif
            end1 = MPI_Wtime();
            time_monitor_YEqn_solve += double(end1 - start1);

            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
    end2 = MPI_Wtime();
    time_monitor_YEqn += double(end2 - start2);
}

Info << "Exit YEqn" << endl;