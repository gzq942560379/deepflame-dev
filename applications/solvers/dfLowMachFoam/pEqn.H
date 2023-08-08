double p_pre_end, p_pre_start, p_pre_time = 0.;
double p_post_end, p_post_start, p_post_time = 0.;
double p_build_end, p_build_start, p_build_time = 0.;
double p_convert_end, p_convert_start, p_convert_time = 0.;
double p_solve_end, p_solve_start, p_solve_time = 0.;
double start, end, time_stick = 0.;


double p_total_start = MPI_Wtime();

p_pre_start = MPI_Wtime();

if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
start = MPI_Wtime();
const volScalarField psip0(psi*p);
end = MPI_Wtime();
Info << "pEqn_stick1.1 = " << end - start << endl;


#ifdef OPT_GenMatrix_p
    start = MPI_Wtime();
    volScalarField rAU = rAUConstructor(UEqn);
    end = MPI_Wtime();
    Info << "pEqn_stick1.2 opt = " << end - start << endl;
#else
    start = MPI_Wtime();
    volScalarField rAU(1.0/UEqn.A());
    end = MPI_Wtime();
    Info << "pEqn_stick1.2 orig = " << end - start << endl;
#endif

start = MPI_Wtime();
volScalarField tmp_rhoRau = rho*rAU;
end = MPI_Wtime();
Info << "multiply time = " << end - start << endl;

#ifdef OPT_GenMatrix_p // FIXME: write intepolation introduce small err
//     start = MPI_Wtime();
//     tmp<surfaceScalarField> trhorAUf = rhorAUfConstructor(tmp_rhoRau, linear_weights);
//     surfaceScalarField rhorAUf = trhorAUf();
//     end = MPI_Wtime();
//     Info << "pEqn_stick1.3 opt = " << end - start << endl;
// #else
    start = MPI_Wtime();
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(tmp_rhoRau));
    end = MPI_Wtime();
    Info << "pEqn_stick1.3 orig = " << end - start << endl;
#endif

#ifdef OPT_GenMatrix_p
    start = MPI_Wtime();
    tmp<volVectorField> tUEqnH = UEqn_H(UEqn);
    end = MPI_Wtime();
    Info << "pEqn_stick1.4 opt = " << end - start << endl;
#else
    start = MPI_Wtime();
    tmp<volVectorField> tUEqnH = UEqn.H();
    end = MPI_Wtime();
    Info << "UEqnH_ref = " << end - start << endl;
#endif

start = MPI_Wtime();
volVectorField HbyA(constrainHbyA(rAU*tUEqnH, U, p));

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}
end = MPI_Wtime();
Info << "pEqn_stick1.5 = " << end - start << endl;

start = MPI_Wtime();
#ifdef OPT_GenMatrix_p
    tmp<surfaceScalarField> tddtCorr = EulerDdtSchemeFvcDdtCorr(rho, U, phi, rhoUf);
    surfaceScalarField ddtCorr = tddtCorr();
#else
    tmp<surfaceScalarField> tddtCorr = fvc::ddtCorr(rho, U, phi, rhoUf);
    surfaceScalarField ddtCorr = tddtCorr();
#endif

// check_field_equal(ddtCorrTest, ddtCorrRef);

#ifdef OPT_GenMatrix_p
//     tmp<surfaceScalarField> tphiHby_test = phiHbyAConstructor(rho, HbyA, rhorAUf, tddtCorr, linear_weights);
//     surfaceScalarField phiHby_test = tphiHby_test.ref();
// #else
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::interpolate(rho)*fvc::flux(HbyA)
        + rhorAUf*tddtCorr
    );
#endif
// construct phiHbyA directly

end = MPI_Wtime();
Info << "pEqn_stick2 opt = " << end - start << endl;

start = MPI_Wtime();
fvc::makeRelative(phiHbyA, rho, U);
end = MPI_Wtime();
Info << "pEqn_stick3 = " << end - start << endl;

// Update the pressure BCs to ensure flux consistency
start = MPI_Wtime();
constrainPressure(p, rho, U, phiHbyA, rhorAUf);
end = MPI_Wtime();
Info << "pEqn_stick4 = " << end - start << endl;

p_pre_end = MPI_Wtime();
p_pre_time += p_pre_end - p_pre_start;

if (pimple.transonic())
{
//     Info << "pimple.transonic()" << endl;
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

        p_convert_start = MPI_Wtime();
#ifdef _CSR_
        csrMatrix csr(pEqn);
#endif
#ifdef _ELL_
        ellMatrix ell(pEqn);
#endif
#ifdef _DIV_
        divMatrix div(pEqn);
#endif
#ifdef _LDU_
        LDUMatrix LDU(pEqn);
#endif
        p_convert_end = MPI_Wtime();
        p_convert_time += p_convert_end - p_convert_start;

        p_solve_start = MPI_Wtime();
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
        p_solve_end = MPI_Wtime();
        p_solve_time += p_solve_end - p_solve_start;

        p_post_start = MPI_Wtime();
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        p_post_end = MPI_Wtime();
        p_post_time += p_post_end - p_post_start;

    }
}
else
{
//     Info << "not pimple.transonic()" << endl;
    p_build_start = MPI_Wtime();
#ifdef OPT_GenMatrix_p
    start = MPI_Wtime();
    tmp<fvScalarMatrix> tpDDtEqn = GenMatrix_p(rho, p, phiHbyA, rhorAUf, psi);
    fvScalarMatrix pDDtEqn = tpDDtEqn.ref();
    end = MPI_Wtime();
    Info << "pEqn_stick5 opt = " << end - start << endl;
#else
    start = MPI_Wtime();
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi * 
        correction(fvm::ddt(p))
        // fvm::ddt(p)
      + fvc::div(phiHbyA)
      - fvm::laplacian(rhorAUf, p)
    );
    end = MPI_Wtime();
    Info << "pEqn_stick5 orig = " << end - start << endl;
#endif

#ifdef OPT_GenMatrix_p_check
    fvScalarMatrix pDDtEqn_answer
    (
        fvc::ddt(rho) + psi * 
        correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
      - fvm::laplacian(rhorAUf, p)
    );

    // check_fvmatrix_equal(pDDtEqn, pDDtEqn_answer);
#endif

    p_build_end = MPI_Wtime();
    p_build_time += p_build_end - p_build_start;

    start = MPI_Wtime();
    while (pimple.correctNonOrthogonal())
    {
        p_build_start = MPI_Wtime();

        fvScalarMatrix pEqn(pDDtEqn);
        p_build_end = MPI_Wtime();
        p_build_time += p_build_end - p_build_start;

        p_convert_start = MPI_Wtime();
#ifdef _CSR_
        csrMatrix csr(pEqn);
#endif
#ifdef _ELL_
        ellMatrix ell(pEqn);
#endif
#ifdef _DIV_
        divMatrix div(pEqn);
#endif
#ifdef _LDU_
        LDUMatrix LDU(pEqn);
#endif
        p_convert_end = MPI_Wtime();
        p_convert_time += p_convert_end - p_convert_start;

        p_solve_start = MPI_Wtime();
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
        p_solve_end = MPI_Wtime();
        p_solve_time += p_solve_end - p_solve_start;

        p_post_start = MPI_Wtime();
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        p_post_end = MPI_Wtime();
        p_post_time += p_post_end - p_post_start;
    }
    end = MPI_Wtime();
    Info << "pEqn_stick6 = " << end - start << endl;
}

p_post_start = MPI_Wtime();

start = MPI_Wtime();
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
end = MPI_Wtime();
Info << "pEqn_stick7 = " << end - start << endl;

start = MPI_Wtime();
U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);
end = MPI_Wtime();
Info << "pEqn_stick8 = " << end - start << endl;

start = MPI_Wtime();
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
end = MPI_Wtime();
Info << "pEqn_stick9 = " << end - start << endl;
p_post_end = MPI_Wtime();
p_post_time += p_post_end - p_post_start;

double p_total_end = MPI_Wtime();
double p_total_time = p_total_end - p_total_start;

Info << "p total Time : " << p_total_time << endl;
Info << "p pre Time : " << p_pre_time << endl;
Info << "p build Time : " << p_build_time << endl;
Info << "p convert Time : " << p_convert_time << endl;
Info << "p solve Time : " << p_solve_time << endl;
Info << "p post Time : " << p_post_time << endl;
Info << "p other Time : " << p_total_time - p_pre_time - p_build_time - p_convert_time - p_solve_time - p_post_time << endl;

