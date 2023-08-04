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
    double GenMatrix_U_tick_0 = MPI_Wtime();

    const fvMesh& mesh = U.mesh();
    assert(mesh.moving() == false);
    
    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();
    
    // basic variable
    double get_weights_begin = MPI_Wtime();
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    double get_weights_end = MPI_Wtime();
    Info << "get_weights : " << get_weights_end - get_weights_begin << endl;


    tmp<fvVectorMatrix> tfvm
    (
        new fvVectorMatrix
        (
            U,
            rho.dimensions()*U.dimensions()*dimVol/dimTime
        )
    );
    fvVectorMatrix& fvm = tfvm.ref();

    double GenMatrix_U_tick_1_1 = MPI_Wtime();
    Info << "GenMatrix_U_time_1_1 : " << GenMatrix_U_tick_1_1 - GenMatrix_U_tick_0 << endl;

    scalar* __restrict__ diagPtr = fvm.diag().begin();
    vector* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ lowerPtr = fvm.lower().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    const labelUList& l = fvm.lduAddr().lowerAddr();
    const labelUList& u = fvm.lduAddr().upperAddr();


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

    double GenMatrix_U_tick_1 = MPI_Wtime();
    Info << "GenMatrix_U_time_1 : " << GenMatrix_U_tick_1 - GenMatrix_U_tick_0 << endl;

    // 1. interpolation
    // TODO: Thread this loop
    for(label f = 0; f < nFaces; ++f){
        pfPtr[f] = weightsPtr[f] * (pPtr[l[f]] - pPtr[u[f]]) + pPtr[u[f]];
    }

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

    double GenMatrix_U_tick_2 = MPI_Wtime();
    Info << "GenMatrix_U_time_2 : " << GenMatrix_U_tick_2 - GenMatrix_U_tick_1 << endl;

    // 2. gradf
    Field<vector>& igGrad = gGrad;
    // TODO: Thread this loop
    for(label facei = 0; facei < nFaces; ++facei){
        vector Sfssf = mesh.Sf()[facei] * pfPtr[facei];
        igGrad[l[facei]] += Sfssf;
        igGrad[u[facei]] -= Sfssf;
    }
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

    double GenMatrix_U_tick_3 = MPI_Wtime();
    Info << "GenMatrix_U_time_3 : " << GenMatrix_U_tick_3 - GenMatrix_U_tick_2 << endl;

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

    double GenMatrix_U_tick_4 = MPI_Wtime();
    Info << "GenMatrix_U_time_4 : " << GenMatrix_U_tick_4 - GenMatrix_U_tick_3 << endl;

    // 1. interpolation
    // TODO: Thread this loop
    for(label f = 0; f < nFaces; ++f){
        Uf[f] = weightsPtr[f] * (UOldTimePtr[l[f]] - UOldTimePtr[u[f]]) + UOldTimePtr[u[f]];
    }

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

    double GenMatrix_U_tick_5 = MPI_Wtime();
    Info << "GenMatrix_U_time_5 : " << GenMatrix_U_tick_5 - GenMatrix_U_tick_4 << endl;

    // 2. gradf
    // face loop to assign cell (conflict)
    Field<tensor>& igGradU = gGradU;
    // TODO: Thread this loop
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

    igGradU /= mesh.V();

    double GenMatrix_U_tick_6 = MPI_Wtime();
    Info << "GenMatrix_U_time_6 : " << GenMatrix_U_tick_6 - GenMatrix_U_tick_5 << endl;

    gGradU.correctBoundaryConditions();

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

    double GenMatrix_U_tick_7 = MPI_Wtime();
    Info << "GenMatrix_U_time_7 : " << GenMatrix_U_tick_7 - GenMatrix_U_tick_6 << endl;

    double GenMatrix_U_tick_7_0 = MPI_Wtime();

    // dev_T
    // tmp<volTensorField> tgGradUCoeff = (turbulence.alpha()*turbulence.rho()*turbulence.nuEff())*dev2(T(gGradU));
    tmp<volTensorField> tgGradUCoeff = (turbulence.rho()*turbulence.nuEff())*dev2(T(gGradU));

    double GenMatrix_U_tick_7_1 = MPI_Wtime();
    Info << "GenMatrix_U_time_7_1 : " << GenMatrix_U_tick_7_1 - GenMatrix_U_tick_7_0 << endl;

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

    double GenMatrix_U_tick_8 = MPI_Wtime();
    Info << "GenMatrix_U_time_8 : " << GenMatrix_U_tick_8 - GenMatrix_U_tick_7 << endl;

    // 1. interpolation
    // TODO: Thread this loop
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

    double GenMatrix_U_tick_9 = MPI_Wtime();
    Info << "GenMatrix_U_time_9 : " << GenMatrix_U_tick_9 - GenMatrix_U_tick_8 << endl;

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
    
    double GenMatrix_U_tick_10 = MPI_Wtime();
    Info << "GenMatrix_U_time_10 : " << GenMatrix_U_tick_10 - GenMatrix_U_tick_9 << endl;

    // TODO: Thread this loop
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

    
    double GenMatrix_U_tick_11 = MPI_Wtime();
    Info << "GenMatrix_U_time_11 : " << GenMatrix_U_tick_11 - GenMatrix_U_tick_10 << endl;

    // igDivGradUCoeff /= mesh.Vsc();
    // gDivGradUCoeff.correctBoundaryConditions();

    // laplacian
    // 1. interpolation
    volScalarField gamma = (turbulence.rho()*turbulence.nuEff()).ref();
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

    double GenMatrix_U_tick_12 = MPI_Wtime();
    Info << "GenMatrix_U_time_12 : " << GenMatrix_U_tick_12 - GenMatrix_U_tick_11 << endl;

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

    double GenMatrix_U_tick_13 = MPI_Wtime();
    Info << "GenMatrix_U_time_13 : " << GenMatrix_U_tick_13 - GenMatrix_U_tick_12 << endl;

    surfaceScalarField gammaMagSf = (gammaf * mesh.magSf()).ref();
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
    for(label c = 0; c < nCells; ++c){
        diagPtr[c] = rDeltaT * rhoPtr[c] * meshVscPtr[c];
        sourcePtr[c] = rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[c] * meshVscPtr[c];
        // grad(p)
        sourcePtr[c] -= igGrad[c];
        // turbulence div
        sourcePtr[c] += igDivGradUCoeff[c];
    }

    for(label f = 0; f < nFaces; ++f){
        lowerPtr[f] = - weightsPtr[f] * phiPtr[f];
        upperPtr[f] = (- weightsPtr[f] + 1.) * phiPtr[f];
        // laplacian
        lowerPtr[f] -= deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        upperPtr[f] -= deltaCoeffsPtr[f] * gammaMagSfPtr[f];
    }

    // TODO: Thread this loop
    for (label face=0; face< nFaces; ++face)
    {
        diagPtr[l[face]] -= lowerPtr[face];
        diagPtr[u[face]] -= upperPtr[face];
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

        fvm.internalCoeffs()[patchi] = patchFlux*psf.valueInternalCoeffs(pw);
        fvm.boundaryCoeffs()[patchi] = -patchFlux*psf.valueBoundaryCoeffs(pw);

        if (psf.coupled())
        {
            fvm.internalCoeffs()[patchi] -=
                pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs);
            fvm.boundaryCoeffs()[patchi] -=
               -pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs);
        }
        else
        {
            fvm.internalCoeffs()[patchi] -= pGamma*psf.gradientInternalCoeffs();
            fvm.boundaryCoeffs()[patchi] -= -pGamma*psf.gradientBoundaryCoeffs();
        }
    }

    double GenMatrix_U_tick_14 = MPI_Wtime();
    Info << "GenMatrix_U_time_14 : " << GenMatrix_U_tick_14 - GenMatrix_U_tick_13 << endl;
 
    return tfvm;
}
}