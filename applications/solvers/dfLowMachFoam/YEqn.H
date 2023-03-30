start = MPI_Wtime();

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

forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
end = MPI_Wtime();
time_monitor_corrDiff += end - start;

{
    if (!splitting)
    {
        start = MPI_Wtime();
        combustion->correct();
        //label flag_mpi_init;
        //MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        end = MPI_Wtime();
        time_monitor_chem += end - start;
    }

    volScalarField Yt(0.0*Y[0]);

    start = MPI_Wtime();
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);

        if (i != inertIndex)
        {
            double Y_solve_total_start = MPI_Wtime();
            double Y_eqn_build_start = MPI_Wtime();
            tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
              + mvConvection->fvmDiv(phi, Yi)
              + mvConvection->fvmDiv(phiUc, Yi)
             ==
                (
                    splitting
                ?   fvm::laplacian(DEff(), Yi)
                :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                )
            );
            double Y_eqn_build_end = MPI_Wtime();

            YiEqn.relax();

            double Y_solve_start = MPI_Wtime();
            YiEqn.solve("Yi");
            double Y_solve_end = MPI_Wtime();

            Yi.max(0.0);
            Yt += Yi;
            double Y_solve_total_end = MPI_Wtime();
            double Y_eqn_build_time = Y_eqn_build_end - Y_eqn_build_start;
            double Y_solve_time = Y_solve_end - Y_solve_start;
            double Y_solve_total_time = Y_solve_total_end - Y_solve_total_start;
            Info << "Y solve_total_time Time : " << Y_solve_total_time << endl;
            Info << "Y eqn_build_time Time : " << Y_eqn_build_time << endl;
            Info << "Y solve_time Time : " << Y_solve_time << endl;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    end = MPI_Wtime();
    time_monitor_Y += end - start;
}
