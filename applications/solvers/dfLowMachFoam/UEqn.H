// Solve the Momentum equation

double U_solve_total_start = MPI_Wtime();
double U_eqn_build_start = MPI_Wtime();
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence->divDevRhoReff(U)
);
double U_eqn_build_end = MPI_Wtime();
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

double U_solve_start = MPI_Wtime();
if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    K = 0.5*magSqr(U);
}
double U_solve_end = MPI_Wtime();
double U_solve_total_end = MPI_Wtime();
double U_eqn_build_time = U_eqn_build_end - U_eqn_build_start;
double U_solve_time = U_solve_end - U_solve_start;
double U_solve_total_time = U_solve_total_end - U_solve_total_start;
Info << "U solve_total_time Time : " << U_solve_total_time << endl;
Info << "U eqn_build_time Time : " << U_eqn_build_time << endl;
Info << "U solve_time Time : " << U_solve_time << endl;