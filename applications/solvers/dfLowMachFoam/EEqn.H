{

    double E_build_start, E_build_end, E_solve_start, E_solve_end, E_convert_start, E_convert_end;
    double E_build_time = 0.;
    double E_convert_time = 0.;
    double E_solve_time = 0.;


    double E_total_start = MPI_Wtime();

    E_build_start = MPI_Wtime();

    volScalarField& he = thermo.he();

    Info << "E equation build scheme : " << endl;
    Info << "fvm::ddt(rho, he) Scheme : " << he.mesh().ddtScheme("ddt(" + he.name() + ')') << endl;
    Info << "fvm::div(phi, he) Scheme : " << he.mesh().divScheme("div(" + phi.name() + ',' + he.name() + ')') << endl;
    Info << "fvc::ddt(rho, K) Scheme : " << K.mesh().ddtScheme("ddt(" + rho.name() + ',' + K.name() + ')') << endl;
    Info << "fvc::div(phi, K) Scheme : " << K.mesh().divScheme("div(" + phi.name() + ',' + K.name() + ')') << endl;
    Info << "fvm::laplacian(turbulence->alphaEff(), he) Scheme : " << he.mesh().laplacianScheme("laplacian(" + turbulence->alphaEff()().name() + ',' + he.name() + ')') << endl;
    Info << "fvc::div(hDiffCorrFlux) Scheme : " << hDiffCorrFlux.mesh().divScheme("div("+hDiffCorrFlux.name()+')') << endl;

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(turbulence->alphaEff(), he)
      + diffAlphaD
     ==
        fvc::div(hDiffCorrFlux)
    );

    EEqn.relax();
    E_build_end = MPI_Wtime();
    E_build_time += E_build_end - E_build_start;

#ifdef _CSR_
    E_convert_start = MPI_Wtime();
    csr.init_value_from_lduMatrix(EEqn);
    E_convert_end = MPI_Wtime();
    E_convert_time += E_convert_end - E_convert_start;
#endif

    E_solve_start = MPI_Wtime();
#ifdef _CSR_
    auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(EEqn.psi());
    auto& source = EEqn.source();
    auto& internalCoeffs = EEqn.internalCoeffs();
    auto& boundaryCoeffs = EEqn.boundaryCoeffs();
    csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
    EEqn.solve();
#endif
    E_solve_end = MPI_Wtime();
    E_solve_time += E_solve_end - E_solve_start;

    double E_total_end = MPI_Wtime();
    double E_total_time = E_total_end - E_total_start;

    Info << "E total Time : " << E_total_time << endl;
    Info << "E build Time : " << E_build_time << endl;
    Info << "E convert Time : " << E_convert_time << endl;
    Info << "E solve Time : " << E_solve_time << endl;
    Info << "E other Time : " << E_total_time - E_build_time - E_convert_time - E_solve_time << endl;

}
