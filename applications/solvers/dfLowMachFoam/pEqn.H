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
getrAUandHbyA(rAU, HbyA, UEqn, U);
#endif

#ifndef OPT_GenMatrix_p
rAU = 1.0/UEqn.A();
HbyA = constrainHbyA(rAU*UEqn.H(), U, p);
rhorAUf = fvc::interpolate(rho*rAU);
#endif

#ifdef OPT_GenMatrix_p_check
tmp<volScalarField> trAUref = 1.0/UEqn.A();
volScalarField& rAUref = trAUref.ref();
tmp<volVectorField> tHbyAref = constrainHbyA(rAUref*UEqn.H(), U, p);
volVectorField& HbyAref = tHbyAref.ref();
tmp<surfaceScalarField> trhorAUfref = fvc::interpolate(rho*rAUref);
surfaceScalarField& rhorAUfref = trhorAUfref.ref();
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

#if defined(_LDU_)
        LDUMatrix LDU(pEqn);
#elif defined(_CSR_)
        csrMatrix csr(pEqn);
#elif defined(_ELL_)
        ellMatrix ell(pEqn);
#elif defined(_DIV_)
        div.copy_value_from_fvMatrix(pEqn);
#else
#endif

        p_convert_time += clock.timeIncrement();

#if defined(_LDU_)
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        LDU.solve(psi, source, internalCoeffs, boundaryCoeffs);
#elif defined(_CSR_)
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#elif defined(_ELL_)
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        ell.solve(psi, source, internalCoeffs, boundaryCoeffs);
#elif defined(_DIV_)
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        div.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
        pEqn.solve();
#endif

        p_solve_time += clock.timeIncrement();

#ifdef OPT_GenMatrix_p
        postProcess_P(p, pEqn, phi, U, rAU, HbyA, K, dpdt, phiHbyA);
#endif

#ifndef OPT_GenMatrix_p
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
#endif

#ifdef OPT_GenMatrix_p_check
        if (pimple.finalNonOrthogonalIter())
        {
            tmp<surfaceScalarField> tphiref = phiHbyARef + pEqn.flux();
            surfaceScalarField& phiref = tphiref.ref();
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

#ifndef OPT_GenMatrix_p
U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);
#endif

#ifdef OPT_GenMatrix_p_check
tmp<volVectorField> tUref = HbyAref - rAUref*fvc::grad(p);
volVectorField& Uref = tUref.ref();
Uref.correctBoundaryConditions();
tmp<volScalarField> tKref = 0.5*magSqr(Uref);
volScalarField& Kref = tKref.ref();
#endif

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

#ifndef OPT_GenMatrix_p
fvc::correctRhoUf(rhoUf, rho, U, phi);
if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdt -= fvc::div(fvc::meshPhi(rho, U), p);
    }
}
#endif

#ifdef OPT_GenMatrix_p_check
if (thermo.dpdt())
{
    tmp<volScalarField> tdpdtref = fvc::ddt(p);
    volScalarField& dpdtref = tdpdtref.ref();

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

