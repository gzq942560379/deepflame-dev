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

{
    {
        start = std::chrono::steady_clock::now();
        chemistry.solve(mesh.time().deltaTValue());
        label flag_mpi_init;
        MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        stop = std::chrono::steady_clock::now();
        processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        time_monitor_chem += processingTime.count();
    }

    volScalarField Yt(0.0*Y[0]);

    start = std::chrono::steady_clock::now();
    forAll(Y, i)
    {
        if (i != inertIndex)
        {
            volScalarField& Yi = Y[i];

            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
             ==
                chemistry.RR(i)
              //+ fvOptions(rho, Yi)
            );

            YiEqn.relax();

            //fvOptions.constrain(YiEqn);

            YiEqn.solve("Yi");

            //fvOptions.correct(Yi);

            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    stop = std::chrono::steady_clock::now();
    processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    time_monitor_Y += processingTime.count();
}