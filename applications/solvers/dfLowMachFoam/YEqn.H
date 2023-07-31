start = MPI_Wtime();

hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;

forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
//label flag_mpi_init;
int flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
end = MPI_Wtime();
time_monitor_corrDiff += end - start;

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

start = MPI_Wtime();

double Y_total_start, Y_total_end, Y_total_time = 0.; 
double Y_init_start, Y_init_end, Y_init_time = 0.; 
double Y_build_start, Y_build_end, Y_build_time = 0.; 
double Y_build_grad_start, Y_build_grad_end, Y_build_grad_time = 0.;  
double Y_build_laplacian_start, Y_build_laplacian_end, Y_build_laplacian_time = 0.;  
double Y_build_opt_start, Y_build_opt_end, Y_build_opt_time = 0.; 
double Y_convert_start, Y_convert_end, Y_convert_time = 0.; 
double Y_update_start, Y_update_end, Y_update_time = 0.; 
double Y_solve_start, Y_solve_end, Y_solve_time = 0.;

double Y_build_old_start, Y_build_old_end, Y_build_old_time = 0.; 
double Y_build_new_start, Y_build_new_end, Y_build_new_time = 0.; 


Y_total_start = MPI_Wtime();

Y_init_start = MPI_Wtime();

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

volScalarField Yt(0.0*Y[0]);

Y_init_end = MPI_Wtime();
Y_init_time += Y_init_end - Y_init_start;
forAll(Y, i)
{
    Y_build_start = MPI_Wtime();
    
    volScalarField& Yi = Y[i];

    auto alpha_mul_hai = thermo.alpha() * chemistry->hai(i);

    // hDiffCorrFlux += chemistry->hai(i) * (chemistry->rhoD(i) * fvc::grad(Yi) - Yi * sumYDiffError);
    Y_build_grad_start = MPI_Wtime();
    auto gradTmp = gaussGradSchemeGrad(Yi);
    Y_build_grad_end = MPI_Wtime();
    Y_build_grad_time += Y_build_grad_end - Y_build_grad_start;

    Y_build_laplacian_start = MPI_Wtime();
    auto laplacianTmp = fvc::laplacian(alpha_mul_hai, Yi);
    Y_build_laplacian_end = MPI_Wtime();
    Y_build_laplacian_time += Y_build_laplacian_end - Y_build_laplacian_start;

    hDiffCorrFlux += chemistry->hai(i) * (chemistry->rhoD(i) * gradTmp - Yi * sumYDiffError);
    diffAlphaD += laplacianTmp;

    Y_build_end = MPI_Wtime();
    Y_build_time += Y_build_end - Y_build_start;
    if (i != inertIndex)
    {

        Y_build_start = MPI_Wtime();
        Y_build_opt_start = MPI_Wtime();
#ifndef OPT_GenMatrix_Y
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
#else
        tmp<fvScalarMatrix> tYiEqn = GenMatrix_Y(rho, Yi, phi, phiUc, chemistry->rhoD(i), turbulence->mut(), splitting, Sct, combustion(), mvConvection.ref());
        fvScalarMatrix& YiEqn = tYiEqn.ref();
#endif
        Y_build_opt_end = MPI_Wtime();
        Y_build_opt_time += Y_build_opt_end - Y_build_opt_start;

        YiEqn.relax();
        Y_build_end = MPI_Wtime();
        Y_build_time += Y_build_end - Y_build_start;
// #ifdef _CSR_
//             Y_convert_start = MPI_Wtime();
//             csr.init_value_from_lduMatrix(YiEqn);
//             Y_convert_end = MPI_Wtime();
//             Y_convert_time += Y_convert_end - Y_convert_start;
// #endif
        Y_solve_start = MPI_Wtime();
// #ifdef _CSR_
//             auto& psi = const_cast<GeometricField<double, fvPatchField, volMesh>&>(YiEqn.psi());
//             auto& source = YiEqn.source();
//             auto& internalCoeffs = YiEqn.internalCoeffs();
//             auto& boundaryCoeffs = YiEqn.boundaryCoeffs();
//             csr.solve(psi, source, internalCoeffs, boundaryCoeffs, "Yi");
// #else
        YiEqn.solve("Yi");
// #endif

        Y_solve_end = MPI_Wtime();
        Y_solve_time += Y_solve_end - Y_solve_start;

        Y_update_start = MPI_Wtime();
        Yi.max(0.0);
        Yt += Yi;
        Y_update_end = MPI_Wtime();
        Y_update_time += Y_update_end - Y_update_start;
    }
}

Y_total_end = MPI_Wtime();
Y_total_time += Y_total_end - Y_total_start;

Info << "Y total Time : " << Y_total_time << endl;
Info << "Y init Time : " << Y_init_time << endl;
Info << "Y build Time : " << Y_build_time << endl;
Info << "Y build opt Time : " << Y_build_opt_time << endl;
Info << "Y build grad Time : " << Y_build_grad_time << endl;
Info << "Y build laplacian Time : " << Y_build_laplacian_time << endl;
Info << "Y convert Time : " << Y_convert_time << endl;
Info << "Y update Time : " << Y_update_time << endl;
Info << "Y solve Time : " << Y_solve_time << endl;
Info << "Y other Time : " << Y_total_time - Y_build_time - Y_convert_time - Y_solve_time - Y_init_time - Y_update_time << endl;

Y[inertIndex] = scalar(1) - Yt;
Y[inertIndex].max(0.0);

end = MPI_Wtime();
time_monitor_Y += end - start;
