// Solve the Momentum equation

double U_build_start, U_build_end, U_build_time = 0.;
double U_convert_start, U_convert_end, U_convert_time = 0.;
double U_solve_start, U_solve_end, U_solve_time = 0.;

double U_build_ddt_start, U_build_ddt_end, U_build_ddt_time = 0.;
double U_build_div_start, U_build_div_end, U_build_div_time = 0.;
double U_build_grad_start, U_build_grad_end, U_build_grad_time = 0.;
double U_build_turbulence_start, U_build_turbulence_end, U_build_turbulence_time = 0.;
double U_build_relax_start, U_build_relax_end, U_build_relax_time = 0.;
double U_build_sum_start, U_build_sum_end, U_build_sum_time = 0.;

Info << "U equation build scheme : " << endl;
Info << "fvm::ddt(rho, U) Scheme : " << U.mesh().ddtScheme("ddt(" + U.name() + ')') << endl;
Info << "fvm::div(phi, U) Scheme : " << U.mesh().divScheme("div(" + phi.name() + ',' + U.name() + ')') << endl;
Info << "fvc::grad(p) Scheme : " << p.mesh().gradScheme("grad(" + p.name() + ')') << endl;

double U_total_start = MPI_Wtime();

U_build_start = MPI_Wtime();
// tmp<fvVectorMatrix> tUEqn
// (
//     fvm::ddt(rho, U) + fvm::div(phi, U)
//   + turbulence->divDevRhoReff(U)
// );

U_build_ddt_start = MPI_Wtime();
tmp<fvVectorMatrix> tUeqnDDT = EulerDdtSchemeFvmDdt(rho, U);
U_build_ddt_end = MPI_Wtime();
U_build_ddt_time += U_build_ddt_end - U_build_ddt_start;

U_build_div_start = MPI_Wtime();
tmp<fvVectorMatrix> tUeqnDIV = gaussConvectionSchemeFvmDiv(phi, U);
U_build_div_end = MPI_Wtime();
U_build_div_time += U_build_div_end - U_build_div_start;

U_build_turbulence_start=MPI_Wtime();
tmp<fvVectorMatrix> tUeqnDivDevRhoReff = turbulence->divDevRhoReff(U);
// tmp<fvVectorMatrix> tUeqnDivDevRhoReff = turbulenceModelLinearViscousStressDivDevRhoReff(U,turbulence());
U_build_turbulence_end=MPI_Wtime();
U_build_turbulence_time += U_build_turbulence_end - U_build_turbulence_start;

U_build_sum_start = MPI_Wtime();
tmp<fvVectorMatrix> tUEqn
(
    tUeqnDDT + tUeqnDIV + tUeqnDivDevRhoReff
);
fvVectorMatrix& UEqn = tUEqn.ref();
U_build_sum_end = MPI_Wtime();
U_build_sum_time += U_build_sum_end - U_build_sum_start;

U_build_relax_start = MPI_Wtime();
UEqn.relax();
U_build_relax_end = MPI_Wtime();
U_build_relax_time += U_build_relax_end - U_build_relax_start;

U_build_end = MPI_Wtime();
U_build_time += U_build_end - U_build_start;
/*
fvVectorMatrix turb_source
(
    turbulence->divDevRhoReff(U)
);*/

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    U_build_start = MPI_Wtime();

    U_build_grad_start = MPI_Wtime();
    auto tGaussGradSchemeGrad = gaussGradSchemeGrad(p);
    U_build_grad_end = MPI_Wtime();
    U_build_grad_time += U_build_grad_end - U_build_grad_start;

    U_build_sum_start = MPI_Wtime();
    fvVectorMatrix UEqnTmp(UEqn == -tGaussGradSchemeGrad);
    fvVectorMatrix& UEqn = tUEqn.ref();
    U_build_sum_end = MPI_Wtime();
    U_build_sum_time += U_build_sum_end - U_build_sum_start;

    U_build_end = MPI_Wtime();
    U_build_time += U_build_end - U_build_start;

#ifdef _CSR_
    U_convert_start = MPI_Wtime();
    csr.init_value_from_lduMatrix(UEqnTmp);
    U_convert_end = MPI_Wtime();
    U_convert_time += U_convert_end - U_convert_start;
#endif


    U_solve_start = MPI_Wtime();
#ifdef _CSR_
    auto& psi = const_cast<GeometricField<vector, fvPatchField, volMesh>&>(UEqnTmp.psi());
    auto& source = UEqnTmp.source();
    auto& internalCoeffs = UEqnTmp.internalCoeffs();
    auto& boundaryCoeffs = UEqnTmp.boundaryCoeffs();
    csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
    UEqnTmp.solve();
#endif

    U_solve_end = MPI_Wtime();
    U_solve_time = U_solve_end - U_solve_start;

    K = 0.5*magSqr(U);
}

double U_total_end = MPI_Wtime();

double U_total_time = U_total_end - U_total_start;

Info << "U total Time : " << U_total_time << endl;
Info << "U build Time : " << U_build_time << endl;
Info << "U build ddt Time : " << U_build_ddt_time << endl;
Info << "U build div Time : " << U_build_div_time << endl;
Info << "U build grad Time : " << U_build_grad_time << endl;
Info << "U build turbulence Time : " << U_build_turbulence_time << endl;
Info << "U build relax Time : " << U_build_relax_time << endl;
Info << "U build sum Time : " << U_build_sum_time << endl;
Info << "U build other Time : " << U_build_time - U_build_ddt_time - U_build_div_time - U_build_grad_time - U_build_turbulence_time - U_build_relax_time - U_build_sum_time << endl;
Info << "U convert Time : " << U_convert_time << endl;
Info << "U solve Time : " << U_solve_time << endl;
Info << "U other Time : " << U_total_time - U_build_time - U_convert_time - U_solve_time << endl;


/*
int offset = 0;
forAll(U.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
    const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];

    const scalarField& patchP = p.boundaryField()[patchi];
    const vectorField& pSf = mesh.Sf().boundaryField()[patchi];

    int patchSize = pw.size();

    Field<vector> ueqn_internalCoeffs_vec = patchFlux*U.boundaryField()[patchi].valueInternalCoeffs(pw);
    Field<vector> ueqn_boundaryCoeffs_vec = -patchFlux*U.boundaryField()[patchi].valueBoundaryCoeffs(pw); 

    // only need to construct once
    std::copy(&ueqn_internalCoeffs_vec[0][0], &ueqn_internalCoeffs_vec[0][0]+3*patchSize, ueqn_internalCoeffs_init + 3*offset);

    // need to construct every time step
    std::copy(&ueqn_boundaryCoeffs_vec[0][0], &ueqn_boundaryCoeffs_vec[0][0]+3*patchSize, ueqn_boundaryCoeffs_init + 3*offset);

    // boundary pressure
    std::copy(&patchP[0], &patchP[0]+patchSize, boundary_pressure_init+offset);

    offset += patchSize;
}
UEqn_csr.fvm_ddt(&rho.oldTime()[0], &rho[0], &U.oldTime()[0][0]);
UEqn_csr.fvm_div(&phi[0], ueqn_internalCoeffs_init, ueqn_boundaryCoeffs_init, boundary_pressure_init);
UEqn_csr.fvc_grad(&p[0]);
UEqn_csr.add_fvMatrix(&turb_source.lower()[0], &turb_source.diag()[0], &turb_source.upper()[0], &turb_source.source()[0][0]);
// UEqn_csr.checkValue(false);*/