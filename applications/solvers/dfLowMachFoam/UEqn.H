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

double U_build_orig_start, U_build_orig_end, U_build_orig_time = 0.;
double U_build_new_start, U_build_new_end, U_build_new_time = 0.;

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
U_build_orig_start = MPI_Wtime();
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U)
    == 
    -fvc::grad(p)
);
U_build_orig_end = MPI_Wtime();
U_build_orig_time = U_build_orig_end - U_build_orig_start;
fvVectorMatrix& UEqn = tUEqn.ref();
U_build_new_start = MPI_Wtime();
tmp<fvVectorMatrix> tUEqn_new = GenMatrix_U(rho, U, phi, p, turbulence());
U_build_new_end = MPI_Wtime();
U_build_new_time = U_build_new_end - U_build_new_start;
fvVectorMatrix& UEqn_new = tUEqn_new.ref();

// assert
assert(UEqn.source().begin() != UEqn_new.source().begin());
for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
{
    scalarField sourceCmpt(UEqn.source().component(cmpt)());
    scalarField sourceCmpt_new(UEqn_new.source().component(cmpt)());
    for (label i = 0; i < sourceCmpt.size(); ++i)
    {
        assert(fabs(sourceCmpt[i] - sourceCmpt_new[i]) < 1e-12);
    }
}

for(label i = 0; i < UEqn.diag().size(); ++i){
    assert(fabs(UEqn.diag()[i] - UEqn_new.diag()[i]) < 1e-12);
}
for(label i = 0; i < UEqn.lower().size(); ++i){
    assert(fabs(UEqn.lower()[i] - UEqn_new.lower()[i]) < 1e-12);
}
for(label i = 0; i < UEqn.upper().size(); ++i){
    assert(fabs(UEqn.upper()[i] - UEqn_new.upper()[i]) < 1e-12);
}
for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
{
    forAll(U.boundaryField(), patchi)
    {
        auto internalCoeffsDiffTmp = UEqn.internalCoeffs().component(cmpt)()[patchi] - UEqn_new.internalCoeffs().component(cmpt)()[patchi];
        auto boundaryCoeffsDiffTmp = UEqn.boundaryCoeffs().component(cmpt)()[patchi] - UEqn_new.boundaryCoeffs().component(cmpt)()[patchi];
        auto& internalCoeffsDiff = internalCoeffsDiffTmp.ref();
        auto& boundaryCoeffsDiff = boundaryCoeffsDiffTmp.ref();

        for(label i = 0; i < internalCoeffsDiff.size(); ++i){
            assert(fabs(internalCoeffsDiff[i]) < 1e-12);
        }
        for(label i = 0; i < UEqn.boundaryCoeffs()[patchi].size(); ++i){
            assert(fabs(boundaryCoeffsDiff[i]) < 1e-12);
        }
    }
}

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
    solve(UEqn_new);

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
    // UEqnTmp.solve();
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
Info << "U build orig Time : " << U_build_orig_time << endl;
Info << "U build new Time : " << U_build_new_time << endl;
Info << "U other Time : " << U_total_time - U_build_time - U_convert_time - U_solve_time << endl;