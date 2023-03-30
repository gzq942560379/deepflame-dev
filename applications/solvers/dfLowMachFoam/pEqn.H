if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

volScalarField rAU(1.0/UEqn.A());
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::interpolate(rho)*fvc::flux(HbyA)
  + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
);

fvc::makeRelative(phiHbyA, rho, U);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);

if (pimple.transonic())
{
    Info << "pimple.transonic()" << endl;
    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);

    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA) + fvm::div(phid, p)
    );

    while (pimple.correctNonOrthogonal())
    {
        double p_solve_total_start = MPI_Wtime();
        double p_eqn_build_start = MPI_Wtime();
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        double p_eqn_build_end = MPI_Wtime();

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();
        
        double p_solve_start = MPI_Wtime();
        pEqn.solve();
        double p_solve_end = MPI_Wtime();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        double p_solve_total_end = MPI_Wtime();
        double p_eqn_build_time = p_eqn_build_end - p_eqn_build_start;
        double p_solve_time = p_solve_end - p_solve_start;
        double p_solve_total_time = p_solve_total_end - p_solve_total_start;
        Info << "p solve_total_time Time : " << p_solve_total_time << endl;
        Info << "p eqn_build_time Time : " << p_eqn_build_time << endl;
        Info << "p solve_time Time : " << p_solve_time << endl;
    }
}
else
{
    Info << "not pimple.transonic()" << endl;
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
    );

    while (pimple.correctNonOrthogonal())
    {
        double p_solve_total_start = MPI_Wtime();
        double p_eqn_build_start = MPI_Wtime();
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        double p_eqn_build_end = MPI_Wtime();

        double p_solve_start = MPI_Wtime();
        pEqn.solve();
        double p_solve_end = MPI_Wtime();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        double p_solve_total_end = MPI_Wtime();
        double p_eqn_build_time = p_eqn_build_end - p_eqn_build_start;
        double p_solve_time = p_solve_end - p_solve_start;
        double p_solve_total_time = p_solve_total_end - p_solve_total_start;
        Info << "p solve_total_time Time : " << p_solve_total_time << endl;
        Info << "p eqn_build_time Time : " << p_eqn_build_time << endl;
        Info << "p solve_time Time : " << p_solve_time << endl;
    }
}

bool limitedp = pressureControl.limit(p);

// Thermodynamic density update
thermo.correctRho(psi*p - psip0);

if (limitedp)
{
    rho = thermo.rho();
}

#include "rhoEqn.H"
#include "compressibleContinuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

// Correct rhoUf if the mesh is moving
fvc::correctRhoUf(rhoUf, rho, U, phi);

if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdt -= fvc::div(fvc::meshPhi(rho, U), p);
    }
}
