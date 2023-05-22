#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

namespace Foam{

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence
){
    double GenMatrix_U_loop_start, GenMatrix_U_loop_end, GenMatrix_U_loop_time = 0.;
    double GenMatrix_U_constructVar_start, GenMatrix_U_constructVar_end, GenMatrix_U_constructVar_time = 0.;
    double GenMatrix_U_turbulence_start, GenMatrix_U_turbulence_end, GenMatrix_U_turbulence_time = 0.;

    const fvMesh& mesh = U.mesh();
    assert(mesh.moving() == false);
    
    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();
    
    // basic variable
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
    tmp<fvVectorMatrix> tfvm_DDT
    (
        new fvVectorMatrix
        (
            U,
            rho.dimensions()*U.dimensions()*dimVol/dimTime
        )
    );
    fvVectorMatrix& fvm_DDT = tfvm_DDT.ref();

    scalar* __restrict__ diagPtr_ddt = fvm_DDT.diag().begin();
    vector* __restrict__ sourcePtr_ddt = fvm_DDT.source().begin();
    scalar* __restrict__ lowerPtr_ddt = fvm_DDT.lower().begin();
    scalar* __restrict__ upperPtr_ddt = fvm_DDT.upper().begin();

    const labelUList& l = fvm_DDT.lduAddr().lowerAddr();
    const labelUList& u = fvm_DDT.lduAddr().upperAddr();


    scalar rDeltaT = 1.0/mesh.time().deltaTValue();
    
    // ddt
    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
    const vector* const __restrict__ UOldTimePtr = U.oldTime().primitiveField().begin();

    // div
    const scalar* const __restrict__ weightsPtr = weights.primitiveField().begin();
    const scalar* const __restrict__ phiPtr = phi.primitiveField().begin();

    // grad
    const scalar* const __restrict__ pPtr = p.primitiveField().begin();
    tmp<surfaceScalarField> tpf(
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+p.name()+')',
                p.instance(),
                p.db()
            ),
            mesh,
            mesh.Sf().dimensions()*p.dimensions()
        )
    );
    surfaceScalarField& pf = tpf.ref();
    scalar* pfPtr = &pf[0];
    tmp<GeometricField<vector, fvPatchField, volMesh>> tgGrad
    (
        new GeometricField<vector, fvPatchField, volMesh>
        (
            IOobject
            (
                "gradP",
                pf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<vector>
            (
                "0",
                pf.dimensions()/dimLength,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<vector>::typeName
        )
    );
    volVectorField& gGrad = tgGrad.ref();

    // 1. interpolation
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for(label f = 0; f < nFaces; ++f){
        pfPtr[f] = weightsPtr[f] * (pPtr[l[f]] - pPtr[u[f]]) + pPtr[u[f]];
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;

    GenMatrix_U_loop_start = MPI_Wtime();
    forAll(weights.boundaryField(), pi)
    {
        const fvsPatchScalarField& pLambda = weights.boundaryField()[pi];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[pi];
        fvsPatchScalarField& psf = pf.boundaryFieldRef()[pi];

        if (p.boundaryField()[pi].coupled())
        {
            psf =
                (
                    pLambda*p.boundaryField()[pi].patchInternalField()
                  + (1.0 - pLambda)*p.boundaryField()[pi].patchNeighbourField()
                );
        }
        else
        {
            psf = p.boundaryField()[pi]; 
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;
    // 2. gradf
    Field<vector>& igGrad = gGrad;
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for(label facei = 0; facei < nFaces; ++facei){
        vector Sfssf = mesh.Sf()[facei] * pfPtr[facei];
        igGrad[l[facei]] += Sfssf;
        igGrad[u[facei]] -= Sfssf;
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;

    GenMatrix_U_loop_start = MPI_Wtime();
    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[patchi];

        const fvsPatchField<scalar>& pssf = pf.boundaryField()[patchi];

        // TODO: Thread this loop
        forAll(mesh.boundary()[patchi], facei)
        {
            igGrad[pFaceCells[facei]] += pSf[facei]*pssf[facei];
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;
    // gGrad.correctBoundaryConditions();
    // igGrad /= mesh.V();

    // ------------------------------------------------------------------------------------------
    // turbulence->divDevRhoReff(U)
    // ------------------------------------------------------------------------------------------
    /**
     * turbulence->divDevRhoReff(U)
     * =
     * - fvc::div((turbulence.alpha()*turbulence.rho()*turbulence.nuEff())*dev2(T(fvc::grad(U))))
     * - fvm::laplacian(turbulence.alpha()*turbulence.rho()*turbulence.nuEff(), U)
     */
    // fvc::div((turbulence.alpha()*turbulence.rho()*turbulence.nuEff())*dev2(T(fvc::grad(U))))
    // gradU
    GenMatrix_U_turbulence_start = MPI_Wtime();
    tmp<surfaceVectorField> tUf(
        new GeometricField<vector, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+U.name()+')',
                U.instance(),
                U.db()
            ),
            mesh,
            mesh.Sf().dimensions()*U.dimensions()
        )
    );
    surfaceVectorField& Uf = tUf.ref();
    tmp<GeometricField<tensor, fvPatchField, volMesh>> tgGradU
    (
        new GeometricField<tensor, fvPatchField, volMesh>
        (
            IOobject
            (
                "gradU",
                Uf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<tensor>
            (
                "0",
                Uf.dimensions()/dimLength,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<tensor>::typeName
        )
    );
    volTensorField& gGradU = tgGradU.ref();
    // 1. interpolation
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for(label f = 0; f < nFaces; ++f){
        Uf[f] = weightsPtr[f] * (UOldTimePtr[l[f]] - UOldTimePtr[u[f]]) + UOldTimePtr[u[f]];
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;

    GenMatrix_U_loop_start = MPI_Wtime();
    forAll(weights.boundaryField(), Ui)
    {
        const fvsPatchScalarField& pLambda = weights.boundaryField()[Ui];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[Ui];
        fvsPatchVectorField& psf = Uf.boundaryFieldRef()[Ui];

        if (U.boundaryField()[Ui].coupled())
        {
            psf =
                (
                    pLambda*U.boundaryField()[Ui].patchInternalField()
                  + (1.0 - pLambda)*U.boundaryField()[Ui].patchNeighbourField()
                );
        }
        else
        {
            psf = U.boundaryField()[Ui]; 
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;
    // 2. gradf
    // face loop to assign cell (conflict)
    Field<tensor>& igGradU = gGradU;
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for(label facei = 0; facei < nFaces; ++facei){
        tensor Sfssf = mesh.Sf()[facei] * Uf[facei];
        igGradU[l[facei]] += Sfssf;
        igGradU[u[facei]] -= Sfssf;
    }
    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[patchi];

        const fvsPatchField<vector>& pssf = Uf.boundaryField()[patchi];

        // TODO: Thread this loop
        forAll(mesh.boundary()[patchi], facei)
        {
            igGradU[pFaceCells[facei]] += pSf[facei]*pssf[facei];
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;

    igGradU /= mesh.V();
    gGradU.correctBoundaryConditions();

    GenMatrix_U_loop_start = MPI_Wtime();
    forAll(U.boundaryField(), patchi)
    {
        fvPatchTensorField& gGradbf = gGradU.boundaryFieldRef()[patchi];

        if (!U.boundaryField()[patchi].coupled())
        {
            const vectorField n
            (
                U.mesh().Sf().boundaryField()[patchi]
              / U.mesh().magSf().boundaryField()[patchi]
            ); //outward normal vector 

            gGradbf += n *
            (
                U.boundaryField()[patchi].snGrad() //snGrad():Surface normal gradient 
              - (n & gGradbf) //projection of gradient of vsf on the boundary onto                                        //the normal vector
            );
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;

    // dev_T
    // tmp<volTensorField> tgGradUCoeff = (turbulence.alpha()*turbulence.rho()*turbulence.nuEff())*dev2(T(gGradU));
    tmp<volTensorField> tgGradUCoeff = (turbulence.rho()*turbulence.nuEff())*dev2(T(gGradU));
    volTensorField gGradUCoeff = tgGradUCoeff.ref();

    // div
    tmp<surfaceVectorField> tgGradUCoeff_f(
        new GeometricField<vector, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+gGradUCoeff.name()+')',
                gGradUCoeff.instance(),
                gGradUCoeff.db()
            ),
            mesh,
            mesh.Sf().dimensions()*gGradUCoeff.dimensions()
        )
    );
    surfaceVectorField& gGradUCoeff_f = tgGradUCoeff_f.ref();
    const tensor* const __restrict__ gGradUCoeffPtr = gGradUCoeff.primitiveField().begin();

    // 1. interpolation
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for (label f = 0; f < nFaces; ++f)
    {
        gGradUCoeff_f[f] = mesh.Sf()[f] & (weightsPtr[f]*(gGradUCoeffPtr[l[f]] - gGradUCoeffPtr[u[f]]) + gGradUCoeffPtr[u[f]]);
    }

    forAll(weights.boundaryField(), gGradUi)
    {
        const fvsPatchScalarField& pLambda = weights.boundaryField()[gGradUi];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[gGradUi];
        fvsPatchVectorField& psf = gGradUCoeff_f.boundaryFieldRef()[gGradUi];

        if (gGradU.boundaryField()[gGradUi].coupled())
        {
            psf =
                pSf
              & (
                    pLambda*gGradUCoeff.boundaryField()[gGradUi].patchInternalField()
                  + (1.0 - pLambda)*gGradUCoeff.boundaryField()[gGradUi].patchNeighbourField()
                );
        }
        else
        {
            psf = pSf & gGradUCoeff.boundaryField()[gGradUi]; 
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;
    // 2. surfaceIntegrate
    tmp<GeometricField<vector, fvPatchField, volMesh>> tgDivGradUCoeff
    (
        new GeometricField<vector, fvPatchField, volMesh>
        (
            IOobject
            (
                "surfaceIntegrate("+gGradUCoeff_f.name()+')',
                gGradUCoeff_f.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<vector>
            (
                "0",
                gGradUCoeff_f.dimensions()/dimVol,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<vector>::typeName
        )
    );
    volVectorField& gDivGradUCoeff = tgDivGradUCoeff.ref();
    vectorField& igDivGradUCoeff = gDivGradUCoeff.primitiveFieldRef();
    
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for (label f = 0; f < nFaces; ++f)
    {
        igDivGradUCoeff[l[f]] += gGradUCoeff_f[f];
        igDivGradUCoeff[u[f]] -= gGradUCoeff_f[f];
    }

    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const fvsPatchField<vector>& pssf = gGradUCoeff_f.boundaryField()[patchi];

        // TODO: Thread this loop
        forAll(mesh.boundary()[patchi], facei)
        {
            igDivGradUCoeff[pFaceCells[facei]] += pssf[facei];
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;
    GenMatrix_U_turbulence_end = MPI_Wtime();
    GenMatrix_U_turbulence_time = GenMatrix_U_turbulence_end - GenMatrix_U_turbulence_start;
    // igDivGradUCoeff /= mesh.Vsc();
    // gDivGradUCoeff.correctBoundaryConditions();

    // laplacian
    // 1. interpolation
    volScalarField gamma = turbulence.rho()*turbulence.nuEff();
    tmp<surfaceScalarField> tgammaf(
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+gamma.name()+')',
                gamma.instance(),
                gamma.db()
            ),
            mesh,
            mesh.Sf().dimensions()*gamma.dimensions()
        )
    );
    surfaceScalarField& gammaf = tgammaf.ref();
    const scalar* const __restrict__ gammaPtr = gamma.primitiveField().begin();
    // TODO: Thread this loop
    GenMatrix_U_loop_start = MPI_Wtime();
    for(label f = 0; f < nFaces; ++f){
        gammaf[f] = weightsPtr[f] * (gammaPtr[l[f]] - gammaPtr[u[f]]) + gammaPtr[u[f]];
    }
    forAll(weights.boundaryField(), gammai)
    {
        const fvsPatchScalarField& pLambda = weights.boundaryField()[gammai];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[gammai];
        fvsPatchScalarField& psf = gammaf.boundaryFieldRef()[gammai];

        if (gamma.boundaryField()[gammai].coupled())
        {
            psf =
                (
                    pLambda*gamma.boundaryField()[gammai].patchInternalField()
                  + (1.0 - pLambda)*gamma.boundaryField()[gammai].patchNeighbourField()
                );
        }
        else
        {
            psf = gamma.boundaryField()[gammai]; 
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;
    surfaceScalarField gammaMagSf = gammaf*mesh.magSf();
    const surfaceScalarField& deltaCoeffs = mesh.nonOrthDeltaCoeffs();

    // fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    // fvm.negSumDiag();

    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();

    // forAll(U.boundaryField(), patchi)
    // {
    //     const fvPatchField<Type>& pvf = U.boundaryField()[patchi];
    //     const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
    //     const fvsPatchScalarField& pDeltaCoeffs =
    //         deltaCoeffs.boundaryField()[patchi];

    //     if (pvf.coupled())
    //     {
    //         fvm.internalCoeffs()[patchi] =
    //             pGamma*pvf.gradientInternalCoeffs(pDeltaCoeffs);
    //         fvm.boundaryCoeffs()[patchi] =
    //            -pGamma*pvf.gradientBoundaryCoeffs(pDeltaCoeffs);
    //     }
    //     else
    //     {
    //         fvm.internalCoeffs()[patchi] = pGamma*pvf.gradientInternalCoeffs();
    //         fvm.boundaryCoeffs()[patchi] = -pGamma*pvf.gradientBoundaryCoeffs();
    //     }
    // }

    // ------------------------------------------------------------------------------------------
    // construct matrix
    // ------------------------------------------------------------------------------------------
    // cell loop
    GenMatrix_U_loop_start = MPI_Wtime();
    #pragma omp parallel for
    #pragma clang loop unroll_count(4)
    #pragma clang loop vectorize(enable)
    for(label c = 0; c < nCells; ++c){
        diagPtr_ddt[c] = rDeltaT * rhoPtr[c] * meshVscPtr[c];
        sourcePtr_ddt[c] = rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[c] * meshVscPtr[c];
        // grad(p)
        sourcePtr_ddt[c] += igGrad[c];
        // turbulence div
        sourcePtr_ddt[c] += igDivGradUCoeff[c];
        // turbulence laplacian

    }

    // face loop to assign face
    #pragma omp parallel for
    #pragma clang loop unroll_count(4)
    #pragma clang loop vectorize(enable)
    for(label f = 0; f < nFaces; ++f){
        lowerPtr_ddt[f] = - weightsPtr[f] * phiPtr[f];
        upperPtr_ddt[f] = (- weightsPtr[f] + 1.) * phiPtr[f];
        // laplacian
        lowerPtr_ddt[f] -= deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        upperPtr_ddt[f] -= deltaCoeffsPtr[f] * gammaMagSfPtr[f];
    }

    // TODO: Thread this loop
    for (label face=0; face< nFaces; ++face)
    {
        diagPtr_ddt[l[face]] -= lowerPtr_ddt[face];
        diagPtr_ddt[u[face]] -= upperPtr_ddt[face];
    }


    // boundary loop
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchField<vector>& psf = U.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs =
            deltaCoeffs.boundaryField()[patchi];

        fvm_DDT.internalCoeffs()[patchi] = patchFlux*psf.valueInternalCoeffs(pw);
        fvm_DDT.boundaryCoeffs()[patchi] = -patchFlux*psf.valueBoundaryCoeffs(pw);

        if (psf.coupled())
        {
            fvm_DDT.internalCoeffs()[patchi] -=
                pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs);
            fvm_DDT.boundaryCoeffs()[patchi] -=
               -pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs);
        }
        else
        {
            fvm_DDT.internalCoeffs()[patchi] -= pGamma*psf.gradientInternalCoeffs();
            fvm_DDT.boundaryCoeffs()[patchi] -= -pGamma*psf.gradientBoundaryCoeffs();
        }
    }
    GenMatrix_U_loop_end = MPI_Wtime();
    GenMatrix_U_loop_time += GenMatrix_U_loop_end - GenMatrix_U_loop_start;

    // construct matrix
    tmp<fvVectorMatrix> tUeqnDIV = gaussConvectionSchemeFvmDiv(phi, U);
    // tmp<fvVectorMatrix> tUeqnDivDevRhoReff = turbulenceModelLinearViscousStressDivDevRhoReff(U,turbulence());
    tmp<fvVectorMatrix> tfvm
    (
        new fvVectorMatrix
        (
            tfvm_DDT
            // + tUeqnDIV 
            // + tUeqnDivDevRhoReff
            // ==
            // -fvc::grad(p)
        )
    );
    Info << "GenMatrix U loop time : " << GenMatrix_U_loop_time << endl;
    Info << "GenMatrix_U_turbulence_time : " << GenMatrix_U_turbulence_time << endl;
    return tfvm;
}
}