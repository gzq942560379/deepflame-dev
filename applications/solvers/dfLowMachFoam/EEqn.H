{

    double E_build_start, E_build_end, E_solve_start, E_solve_end, E_convert_start, E_convert_end;
    double E_build_time = 0.;
    double E_convert_time = 0.;
    double E_solve_time = 0.;
    
    double E_build_org_start, E_build_org_end, E_build_org_time = 0.;
    double E_build_opt_start, E_build_opt_end, E_build_opt_time = 0.;


    double E_total_start = MPI_Wtime();

    E_build_start = MPI_Wtime();

    volScalarField& he = thermo.he();

    volScalarField turbulence_alphaEff
    (
        IOobject
        (
            "turbulence_alphaEff",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1,-1,-1,0,0,0,0), 0)
    );

    turbulence_alphaEff = turbulence->alphaEff();

#ifndef OPT_GenMatrix_E
    E_build_org_start = MPI_Wtime();
    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
        + fvc::ddt(rho, K) + fvc::div(phi, K)
        - dpdt
        - fvm::laplacian(turbulence_alphaEff, he)
        + diffAlphaD
         ==
            fvc::div(hDiffCorrFlux)
    );
    E_build_org_end = MPI_Wtime();
    Info << "E_build_org_time = " << E_build_org_end - E_build_org_start << endl;
#else
    E_build_opt_start = MPI_Wtime();
    tmp<fvScalarMatrix> tEEqn = GenMatrix_E(rho, he, phi, K, dpdt, turbulence_alphaEff, diffAlphaD, hDiffCorrFlux, linear_weights, face_scheduling);
    fvScalarMatrix& EEqn = tEEqn.ref();
    E_build_opt_end = MPI_Wtime();
    Info << "E_build_opt_time = " << E_build_opt_end - E_build_opt_start << endl;
#endif

    EEqn.relax();
    E_build_end = MPI_Wtime();
    E_build_time += E_build_end - E_build_start;

#ifdef OPT_GenMatrix_E_check
    E_build_org_start = MPI_Wtime();
    fvScalarMatrix EEqn_answer
    (
        fvm::ddt(rho, he) 
        + fvm::div(phi, he)
        + fvc::ddt(rho, K) 
        + fvc::div(phi, K)
        - dpdt
        - fvm::laplacian(turbulence->alphaEff(), he)
        + diffAlphaD
         ==
            fvc::div(hDiffCorrFlux)
    );
    E_build_org_end = MPI_Wtime();
    Info << "E_build_org_time = " << E_build_org_end - E_build_org_start << endl;
    check_fvmatrix_equal(EEqn, EEqn_answer);
#endif

// #ifdef _CSR_
//     E_convert_start = MPI_Wtime();
//     csr.init_value_from_lduMatrix(EEqn);
//     E_convert_end = MPI_Wtime();
//     E_convert_time += E_convert_end - E_convert_start;
// #endif

    E_solve_start = MPI_Wtime();
// #ifdef _CSR_
//     auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(EEqn.psi());
//     auto& source = EEqn.source();
//     auto& internalCoeffs = EEqn.internalCoeffs();
//     auto& boundaryCoeffs = EEqn.boundaryCoeffs();
//     csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
// #else
    EEqn.solve();
// #endif
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
