double p_build_opt_start, p_build_opt_end, p_build_opt_time = 0.;
double p_build_org_start, p_build_org_end, p_build_org_time = 0.;

double p_pre_time = 0.;
double p_post_time = 0.;
double p_build_time = 0.;
double p_convert_time = 0.;
double p_solve_time = 0.;
double p_pimple_time = 0.;

clockTime clock;

if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

#ifdef OPT_GenMatrix_p
// - get rAU
// getrAU(rAU)
// - get HbyA
// getHbyA(HbyA)
getrAUandHbyA(rAU, HbyA, UEqn, U);
#endif

rAU = 1.0/UEqn.A(); // TODO: optimize
HbyA = constrainHbyA(rAU*UEqn.H(), U, p); // TODO: optimize
#ifndef OPT_GenMatrix_p
rhorAUf = fvc::interpolate(rho*rAU);
#endif

#ifdef OPT_GenMatrix_p_check
volScalarField rAUref = 1.0/UEqn.A();
volVectorField HbyAref = constrainHbyA(rAUref*UEqn.H(), U, p);
surfaceScalarField rhorAUfref = fvc::interpolate(rho*rAUref);
check_field_boundary_equal(rAUref, rAU, "rAU");
check_field_boundary_equal(HbyAref, HbyA, "HbyA");
#endif

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}

#ifndef OPT_GenMatrix_p
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::interpolate(rho)*fvc::flux(HbyA)
    + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
);
#endif

#ifdef OPT_GenMatrix_p_check
surfaceScalarField phiHbyARef
(
    "phiHbyARef",
    fvc::interpolate(rho)*fvc::flux(HbyAref)
    + rhorAUfref*fvc::ddtCorr(rho, U, phi, rhoUf)
);
#endif

p_pre_time += clock.timeIncrement();

// {
    p_pimple_time += clock.timeIncrement();

#ifdef OPT_GenMatrix_p
    p_build_opt_start = MPI_Wtime();
    tmp<fvScalarMatrix> tpDDtEqn = GenMatrix_p(rho, p, phi, rAU, U, HbyA, psi, rhorAUf, phiHbyA);

    fvScalarMatrix pDDtEqn = tpDDtEqn.ref();
    p_build_opt_end = MPI_Wtime();
    p_build_opt_time += p_build_opt_end - p_build_opt_start;
    Info << "p_build_opt_time : " << p_build_opt_time << endl;
#else
    p_build_org_start = MPI_Wtime();
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
      + fvc::div(phiHbyA) - fvm::laplacian(rhorAUf, p)
    );
    p_build_org_end = MPI_Wtime();
    p_build_org_time += p_build_org_end - p_build_org_start;
    Info << "p_build_org_time : " << p_build_org_time << endl;
#endif

#ifdef OPT_GenMatrix_p_check
    p_build_org_start = MPI_Wtime();
    fvScalarMatrix pDDtEqn_answer
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
      + fvc::div(phiHbyARef)
      - fvm::laplacian(rhorAUfref, p)
    );
    p_build_org_end = MPI_Wtime();
    p_build_org_time += p_build_org_end - p_build_org_start;
    Info << "p_build_org_time : " << p_build_org_time << endl;
    check_fvmatrix_equal(pDDtEqn, pDDtEqn_answer, "pDDtEqn");
#endif

    p_build_time += clock.timeIncrement();

    while (pimple.correctNonOrthogonal())
    {
        p_pimple_time += clock.timeIncrement();

        fvScalarMatrix pEqn(pDDtEqn);

        p_build_time += clock.timeIncrement();

#ifdef _CSR_
        csrMatrix csr(pEqn);
#endif
#ifdef _ELL_
        ellMatrix ell(pEqn);
#endif
#ifdef _DIV_
        // divMatrix div(pEqn);
        div.copy_value_from_fvMatrix(pEqn);
#endif
#ifdef _LDU_
        LDUMatrix LDU(pEqn);
#endif
        p_convert_time += clock.timeIncrement();

#ifdef _CSR_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _ELL_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        ell.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _DIV_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        div.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _LDU_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        LDU.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif

#if !defined(_CSR_) && !defined(_ELL_) && !defined(_LDU_) && !defined(_DIV_)
        pEqn.solve();
#endif
        p_solve_time += clock.timeIncrement();

        postProcess_P(p, pEqn, phi, U, rAU, HbyA, K, dpdt, phiHbyA);

#ifdef OPT_GenMatrix_p_check
        if (pimple.finalNonOrthogonalIter())
        {
            surfaceScalarField phiref = phiHbyARef + pEqn.flux();
            check_field_boundary_equal(phiref, phi, "phi");
        }
#endif
        p_post_time += clock.timeIncrement();
    }
// }

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

#ifdef OPT_GenMatrix_p_check
volVectorField Uref = HbyAref - rAUref*fvc::grad(p);
Uref.correctBoundaryConditions();
volScalarField Kref = 0.5*magSqr(Uref);
#endif

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

#ifdef OPT_GenMatrix_p_check
if (thermo.dpdt())
{
    volScalarField dpdtref = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdtref -= fvc::div(fvc::meshPhi(rho, U), p);
    }

    check_field_equal(dpdtref, dpdt, "dpdt");
}
check_field_boundary_equal(Uref, U, "U");
check_field_boundary_equal(Kref, K, "K");
#endif

p_post_time += clock.timeIncrement();

double p_total_time = clock.elapsedTime();

Info << "p total Time : " << p_total_time << endl;
Info << "p pre Time : " << p_pre_time << endl;
Info << "p build Time : " << p_build_time << endl;
Info << "p convert Time : " << p_convert_time << endl;
Info << "p solve Time : " << p_solve_time << endl;
Info << "p post Time : " << p_post_time << endl;
Info << "p p_pimple_time Time : " << p_pimple_time << endl;
Info << "p other Time : " << p_total_time - p_pre_time - p_build_time - p_convert_time - p_solve_time - p_post_time - p_pimple_time << endl;

