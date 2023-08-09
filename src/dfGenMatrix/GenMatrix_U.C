#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

// #define TIME
#ifdef TIME 
#define TICK0(prefix)\
    double prefix##_tick_0 = MPI_Wtime();

#define TICK(prefix,start,end)\
    double prefix##_tick_##end = MPI_Wtime();\
    Info << #prefix << "_time_" << #end << " : " << prefix##_tick_##end - prefix##_tick_##start << endl;

#else
#define TICK0(prefix) ;
#define TICK(prefix,start,end) ;
#endif
namespace Foam{

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence,
    labelList& face_scheduling
){
    TICK0(GenMatrix_U);

    const fvMesh& mesh = U.mesh(); 
    assert(mesh.moving() == false);
    
    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();

    Info << "nCells : " << nCells << endl;
    Info << "nFaces : " << nFaces << endl;

    // basic variable
    // expansive but only once
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const surfaceScalarField& deltaCoeffs = mesh.nonOrthDeltaCoeffs();

    TICK(GenMatrix_U, 0 , 1);

    tmp<fvVectorMatrix> tfvm
    (
        new fvVectorMatrix
        (
            U,
            rho.dimensions()*U.dimensions()*dimVol/dimTime
        )
    );
    fvVectorMatrix& fvm = tfvm.ref();

    const labelUList& l = fvm.lduAddr().lowerAddr();
    const labelUList& u = fvm.lduAddr().upperAddr();


    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    TICK(GenMatrix_U, 1, 2);

    surfaceScalarField pf(
        IOobject
        (
            "interpolate("+p.name()+')',
            p.instance(),
            p.db()
        ),
        mesh,
        mesh.Sf().dimensions()*p.dimensions()
    );
    surfaceVectorField Uf(
        IOobject
        (
            "interpolate("+U.name()+')',
            U.instance(),
            U.db()
        ),
        mesh,
        mesh.Sf().dimensions()*U.dimensions()
    );

    TICK(GenMatrix_U, 2, 3);
 
    // TODO
    volScalarField gamma = (turbulence.rho()*turbulence.nuEff()).ref();
   
    TICK(GenMatrix_U, 3, 4);
    surfaceScalarField gammaf(
        IOobject
        (
            "interpolate("+gamma.name()+')',
            gamma.instance(),
            gamma.db()
        ),
        mesh,
        mesh.Sf().dimensions()*gamma.dimensions()
    );
    TICK(GenMatrix_U, 4, 5);


    scalar* pfPtr = pf.begin();
    const vector* const __restrict__ UOldTimePtr = U.oldTime().primitiveField().begin();
    const scalar* const __restrict__ weightsPtr = weights.primitiveField().begin();
    const scalar* const __restrict__ phiPtr = phi.primitiveField().begin();
    const scalar* const __restrict__ pPtr = p.primitiveField().begin();
    const scalar* const __restrict__ gammaPtr = gamma.primitiveField().begin();

    #pragma omp parallel for
    for(label facei = 0; facei < nFaces; ++facei){
        pfPtr[facei] = weightsPtr[facei] * (pPtr[l[facei]] - pPtr[u[facei]]) + pPtr[u[facei]];
        Uf[facei] = weightsPtr[facei] * (UOldTimePtr[l[facei]] - UOldTimePtr[u[facei]]) + UOldTimePtr[u[facei]];
        gammaf[facei] = weightsPtr[facei] * (gammaPtr[l[facei]] - gammaPtr[u[facei]]) + gammaPtr[u[facei]];
    }

    TICK(GenMatrix_U, 5, 6);

    forAll(weights.boundaryField(), wi)
    {
        const fvsPatchScalarField& pLambda = weights.boundaryField()[wi];
        fvsPatchScalarField& ppsf = pf.boundaryFieldRef()[wi];
        fvsPatchVectorField& Upsf = Uf.boundaryFieldRef()[wi];
        
        assert((p.boundaryField()[wi].coupled() && U.boundaryField()[wi].coupled()) || (!p.boundaryField()[wi].coupled() && !U.boundaryField()[wi].coupled()));

        if (p.boundaryField()[wi].coupled())
        {
            ppsf = pLambda * p.boundaryField()[wi].patchInternalField() + (1.0 - pLambda) * p.boundaryField()[wi].patchNeighbourField();
            Upsf = pLambda * U.boundaryField()[wi].patchInternalField() + (1.0 - pLambda) * U.boundaryField()[wi].patchNeighbourField();
        }
        else
        {
            ppsf = p.boundaryField()[wi]; 
            Upsf = U.boundaryField()[wi]; 
        }

    }
    TICK(GenMatrix_U, 6, 7);

    volVectorField gGrad(
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
    );
    volTensorField gGradU(
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
    );
    Field<vector>& igGrad = gGrad;
    Field<tensor>& igGradU = gGradU;

    #pragma omp parallel for
    for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label facei = face_start; facei < face_end; ++facei){
            vector pSfssf = mesh.Sf()[facei] * pfPtr[facei];
            igGrad[l[facei]] += pSfssf;
            igGrad[u[facei]] -= pSfssf;
            tensor USfssf = mesh.Sf()[facei] * Uf[facei];
            igGradU[l[facei]] += USfssf;
            igGradU[u[facei]] -= USfssf;
        }
    }
    #pragma omp parallel for
    for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label facei = face_start; facei < face_end; ++facei){
            vector pSfssf = mesh.Sf()[facei] * pfPtr[facei];
            igGrad[l[facei]] += pSfssf;
            igGrad[u[facei]] -= pSfssf;
            tensor USfssf = mesh.Sf()[facei] * Uf[facei];
            igGradU[l[facei]] += USfssf;
            igGradU[u[facei]] -= USfssf;
        }
    }

    TICK(GenMatrix_U, 7, 8);

    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();

        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[patchi];

        const fvsPatchField<scalar>& ppssf = pf.boundaryField()[patchi];
        const fvsPatchField<vector>& Upssf = Uf.boundaryField()[patchi];

        // TODO: Thread this loop
        forAll(mesh.boundary()[patchi], facei)
        {
            igGrad[pFaceCells[facei]] += pSf[facei]*ppssf[facei];
            igGradU[pFaceCells[facei]] += pSf[facei]*Upssf[facei];
        }
    }
    TICK(GenMatrix_U, 8, 9);

    // igGradU /= mesh.V();
    
    #pragma omp parallel for
    for(label c = 0; c < nCells; ++c){
        igGradU[c] /= mesh.V()[c];
    }
    TICK(GenMatrix_U, 9, 10);

    // TODO
    gGradU.correctBoundaryConditions();

    TICK(GenMatrix_U, 10, 11);

    forAll(U.boundaryField(), patchi)
    {
        fvPatchTensorField& gGradbf = gGradU.boundaryFieldRef()[patchi];

        if (!U.boundaryField()[patchi].coupled())
        {
            const vectorField n(U.mesh().Sf().boundaryField()[patchi] / U.mesh().magSf().boundaryField()[patchi]); //outward normal vector 
            //snGrad():Surface normal gradient 
            //projection of gradient of vsf on the boundary onto                                        //the normal vector
            gGradbf += n * (U.boundaryField()[patchi].snGrad() - (n & gGradbf));
        }
    }

    TICK(GenMatrix_U, 11, 12);

    // TODO
    tmp<volTensorField> tgGradUCoeff = (turbulence.rho()*turbulence.nuEff())*dev2(T(gGradU));
    volTensorField gGradUCoeff = tgGradUCoeff.ref();

    TICK(GenMatrix_U, 12, 13);

    surfaceVectorField gGradUCoeff_f(
        IOobject
        (
            "interpolate("+gGradUCoeff.name()+')',
            gGradUCoeff.instance(),
            gGradUCoeff.db()
        ),
        mesh,
        mesh.Sf().dimensions()*gGradUCoeff.dimensions()
    );

    volVectorField gDivGradUCoeff(
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
    );
    vectorField& igDivGradUCoeff = gDivGradUCoeff.primitiveFieldRef();

    const tensor* const __restrict__ gGradUCoeffPtr = gGradUCoeff.primitiveField().begin();

    TICK(GenMatrix_U, 13, 14);

    #pragma omp parallel for
    for (label facei = 0; facei < nFaces; ++facei)
    {
        gGradUCoeff_f[facei] = mesh.Sf()[facei] & (weightsPtr[facei]*(gGradUCoeffPtr[l[facei]] - gGradUCoeffPtr[u[facei]]) + gGradUCoeffPtr[u[facei]]);
    }

    // for (label facei = 0; facei < nFaces; ++facei)
    // {
    //     igDivGradUCoeff[l[facei]] += gGradUCoeff_f[facei];
    //     igDivGradUCoeff[u[facei]] -= gGradUCoeff_f[facei];
    // }
    
    #pragma omp parallel for
    for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label facei = face_start; facei < face_end; ++facei){
            igDivGradUCoeff[l[facei]] += gGradUCoeff_f[facei];
            igDivGradUCoeff[u[facei]] -= gGradUCoeff_f[facei];
        }
    }
    #pragma omp parallel for
    for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label facei = face_start; facei < face_end; ++facei){
            igDivGradUCoeff[l[facei]] += gGradUCoeff_f[facei];
            igDivGradUCoeff[u[facei]] -= gGradUCoeff_f[facei];
        }
    }

    TICK(GenMatrix_U, 14, 15);

    forAll(weights.boundaryField(), wi)
    {
        const fvsPatchScalarField& pLambda = weights.boundaryField()[wi];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[wi];
        fvsPatchVectorField& gGradUCoeff_f_psf = gGradUCoeff_f.boundaryFieldRef()[wi];
        fvsPatchScalarField& gammaf_psf = gammaf.boundaryFieldRef()[wi];

        assert((gGradU.boundaryField()[wi].coupled() && gamma.boundaryField()[wi].coupled()) || (!gGradU.boundaryField()[wi].coupled() && !gamma.boundaryField()[wi].coupled()));

        if (gGradU.boundaryField()[wi].coupled())
        {
            gGradUCoeff_f_psf = pSf & (pLambda*gGradUCoeff.boundaryField()[wi].patchInternalField() + (1.0 - pLambda) * gGradUCoeff.boundaryField()[wi].patchNeighbourField());
            gammaf_psf = (pLambda*gamma.boundaryField()[wi].patchInternalField() + (1.0 - pLambda)*gamma.boundaryField()[wi].patchNeighbourField());
        }
        else
        {
            gGradUCoeff_f_psf = pSf & gGradUCoeff.boundaryField()[wi]; 
            gammaf_psf = gamma.boundaryField()[wi]; 
        }
    }

    TICK(GenMatrix_U, 15, 16);


    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();

        const fvsPatchField<vector>& pssf = gGradUCoeff_f.boundaryField()[patchi];

        forAll(mesh.boundary()[patchi], facei)
        {
            igDivGradUCoeff[pFaceCells[facei]] += pssf[facei];
        }
    }

    TICK(GenMatrix_U, 16, 17);

    surfaceScalarField gammaMagSf = (gammaf * mesh.magSf()).ref();

    TICK(GenMatrix_U, 17, 18);

    TICK(GenMatrix_U, 18, 19);

    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();

    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();

    scalar* __restrict__ diagPtr = fvm.diag().begin();
    vector* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ lowerPtr = fvm.lower().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    TICK(GenMatrix_U, 19, 20);

    #pragma omp parallel for
    for(label c = 0; c < nCells; ++c){
        diagPtr[c] = rDeltaT * rhoPtr[c] * meshVscPtr[c];
        sourcePtr[c] = rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[c] * meshVscPtr[c];
        sourcePtr[c] -= igGrad[c];
        sourcePtr[c] += igDivGradUCoeff[c];
    }
    TICK(GenMatrix_U, 20, 21);

    #pragma omp parallel for
    for(label f = 0; f < nFaces; ++f){
        lowerPtr[f] = - weightsPtr[f] * phiPtr[f];
        upperPtr[f] = (- weightsPtr[f] + 1.) * phiPtr[f];
        lowerPtr[f] -= deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        upperPtr[f] -= deltaCoeffsPtr[f] * gammaMagSfPtr[f];
    }
    TICK(GenMatrix_U, 21, 22);

    // for (label face=0; face< nFaces; ++face)
    // {
    //     diagPtr[l[face]] -= lowerPtr[face];
    //     diagPtr[u[face]] -= upperPtr[face];
    // }
    #pragma omp parallel for
    for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label facei = face_start; facei < face_end; ++facei){
            diagPtr[l[facei]] -= lowerPtr[facei];
            diagPtr[u[facei]] -= upperPtr[facei];
        }
    }
    #pragma omp parallel for
    for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label facei = face_start; facei < face_end; ++facei){
            diagPtr[l[facei]] -= lowerPtr[facei];
            diagPtr[u[facei]] -= upperPtr[facei];
        }
    }

    TICK(GenMatrix_U, 22, 23);

    forAll(U.boundaryField(), patchi)
    {
        const fvPatchField<vector>& Upsf = U.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs = deltaCoeffs.boundaryField()[patchi];

        fvm.internalCoeffs()[patchi] = patchFlux * Upsf.valueInternalCoeffs(pw);
        fvm.boundaryCoeffs()[patchi] = -patchFlux * Upsf.valueBoundaryCoeffs(pw);

        if (Upsf.coupled())
        {
            fvm.internalCoeffs()[patchi] -= pGamma * Upsf.gradientInternalCoeffs(pDeltaCoeffs);
            fvm.boundaryCoeffs()[patchi] -= -pGamma * Upsf.gradientBoundaryCoeffs(pDeltaCoeffs);
        }
        else
        {
            fvm.internalCoeffs()[patchi] -= pGamma * Upsf.gradientInternalCoeffs();
            fvm.boundaryCoeffs()[patchi] -= -pGamma * Upsf.gradientBoundaryCoeffs();
        }
    }
    TICK(GenMatrix_U, 23, 24);

    return tfvm;
}
}