{
    volScalarField& he = thermo.he();
    if (constProp == "volume") he[0] = u0 + p[0]/rho[0];
    start = MPI_Wtime();
    chemistry.correctThermo();
    end = MPI_Wtime();
    time_monitor_corrThermo += end - start;
}
