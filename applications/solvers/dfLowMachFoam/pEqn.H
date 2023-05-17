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

double p_build_end, p_build_start;
double p_convert_end, p_convert_start;
double p_solve_end, p_solve_start;

double p_build_time = 0.;
double p_solve_time = 0.;
double p_convert_time = 0.;

double p_total_start = MPI_Wtime();

if (pimple.transonic())
{
    Info << "pimple.transonic()" << endl;
    p_build_start = MPI_Wtime();

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
    p_build_end = MPI_Wtime();
    p_build_time += p_build_end - p_build_start;

    while (pimple.correctNonOrthogonal())
    {
        p_build_start = MPI_Wtime();
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();
        p_build_end = MPI_Wtime();
        p_build_time += p_build_end - p_build_start;


#ifdef _CSR_
        p_convert_start = MPI_Wtime();
        csr.init_value_from_lduMatrix(pEqn);
        p_convert_end = MPI_Wtime();
        p_convert_time += p_convert_end - p_convert_start;
#endif   

        p_solve_start = MPI_Wtime();
#ifdef _CSR_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
        pEqn.solve();
#endif
        p_solve_end = MPI_Wtime();
        p_solve_time += p_solve_end - p_solve_start;

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    Info << "not pimple.transonic()" << endl;

    Info << "p equation build scheme : " << endl;
    Info << "fvc::ddt(rho) Scheme : " << rho.mesh().ddtScheme("ddt(" + rho.name() + ')') << endl;
    Info << "fvm::ddt(p) Scheme : " << p.mesh().ddtScheme("ddt(" + p.name() + ')') << endl;
    // Info << "fvc::div(phiHbyA) Scheme : " <<  << endl;
    Info << "fvm::laplacian(rhorAUf, p) Scheme : " << p.mesh().laplacianScheme("laplacian(" + rhorAUf.name() + ',' + p.name() + ')') << endl;
    
    p_build_start = MPI_Wtime();
    
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
    );

    p_build_end = MPI_Wtime();
    p_build_time += p_build_end - p_build_start;

    while (pimple.correctNonOrthogonal())
    {
        p_build_start = MPI_Wtime();

        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        p_build_end = MPI_Wtime();
        p_build_time += p_build_end - p_build_start;

#ifdef _CSR_
        p_convert_start = MPI_Wtime();
        csr.init_value_from_lduMatrix(pEqn);
        p_convert_end = MPI_Wtime();
        p_convert_time += p_convert_end - p_convert_start;
#endif   

        p_solve_start = MPI_Wtime();
#ifdef _CSR_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
        pEqn.solve();
#endif
        p_solve_end = MPI_Wtime();
        p_solve_time += p_solve_end - p_solve_start;

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}

double p_total_end = MPI_Wtime();
double p_total_time = p_total_end - p_total_start;

Info << "p total Time : " << p_total_time << endl;
Info << "p build Time : " << p_build_time << endl;
Info << "p convert Time : " << p_convert_time << endl;
Info << "p solve Time : " << p_solve_time << endl;
Info << "p other Time : " << p_total_time - p_build_time - p_convert_time - p_solve_time << endl;

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
