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
        start = MPI_Wtime();
        chemistry.solve(mesh.time().deltaTValue());
        //label flag_mpi_init;
        int flag_mpi_init;
        MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        end = MPI_Wtime();
        time_monitor_chem += end - start;
    }

    volScalarField Yt(0.0*Y[0]);

    start = MPI_Wtime();
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

    end = MPI_Wtime();
    time_monitor_Y += end - start;
}