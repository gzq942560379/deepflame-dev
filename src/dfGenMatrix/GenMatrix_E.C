#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

namespace Foam{

tmp<fvScalarMatrix>
GenMatrix_E(
    const volScalarField& rho,
    volScalarField& he,
    const surfaceScalarField& phi,
    const volScalarField& K,
    const volScalarField& dpdt,
    const volScalarField& alphaEff,
    const volScalarField& diffAlphaD,
    const volVectorField& hDiffCorrFlux,
    const surfaceScalarField& linear_weights
){
    double total_begin = MPI_Wtime();

    const fvMesh& mesh = he.mesh();
    assert(mesh.moving() == false);

    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();

    // basic matrix
    double time_begin = MPI_Wtime();
    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            he,
            rho.dimensions()*he.dimensions()*dimVol/dimTime
        )
    );
    fvScalarMatrix& fvm = tfvm.ref();

    scalar* __restrict__ diagPtr = fvm.diag().begin();
    scalar* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ lowerPtr = fvm.lower().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    const labelUList& l = fvm.lduAddr().lowerAddr();
    const labelUList& u = fvm.lduAddr().upperAddr();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();
    double time_end = MPI_Wtime();
    Info << "prepareTime = " << time_end - time_begin << endl;


    // fvmddt
    time_begin = MPI_Wtime();
    
    // auto fvmDdtTmp = EulerDdtSchemeFvmDdt(rho, he);

    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ meshVPtr = mesh.V().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
    const scalar* const __restrict__ heOldTimePtr = he.oldTime().primitiveField().begin();

    time_end = MPI_Wtime();
    Info << "fvmDdtTime = " << time_end - time_begin << endl;

    // fvmdiv
    time_begin = MPI_Wtime();

    // auto fvmDivTmp = gaussConvectionSchemeFvmDiv(phi, he);


    // - get weights TODO:time consuming
    tmp<fv::convectionScheme<scalar>> cs = 
        fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div("+phi.name()+','+he.name()+')'));
    fv::gaussConvectionScheme<scalar>& gcs = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs.ref());

    tmp<surfaceScalarField> tweights = gcs.interpScheme().weights(he);
    const surfaceScalarField& weights = tweights();

    // - pointers
    // const scalar* const __restrict__ weightsPtr = linear_weights.primitiveField().begin(); // get linear weight
    const scalar* const __restrict__ weightsPtr = weights.primitiveField().begin(); // get weight
    const scalar* const __restrict__ phiPtr = phi.primitiveField().begin();


    time_end = MPI_Wtime();
    Info << "fvmDivTime = " << time_end - time_begin << endl;

    // fvcdiv
    time_begin = MPI_Wtime();

    // auto fvcDivTmp = gaussConvectionSchemeFvcDiv(phi, K);

    // - interpolation
    tmp<surfaceScalarField> tKf(
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+K.name()+')',
                K.instance(),
                K.db()
            ),
            mesh,
            mesh.Sf().dimensions()*K.dimensions()
        )
    );
    surfaceScalarField& Kf = tKf.ref();
    scalar* KfPtr = &Kf[0];
    const scalar* const __restrict__ KPtr = K.primitiveField().begin();

    for(label f = 0; f < nFaces; ++f){
        KfPtr[f] = weightsPtr[f] * (KPtr[l[f]] - KPtr[u[f]]) + KPtr[u[f]];
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[Ki];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[Ki];
        fvsPatchScalarField& psf = Kf.boundaryFieldRef()[Ki];

        if (K.boundaryField()[Ki].coupled())
        {
            psf =
                (
                    pLambda*K.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*K.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = K.boundaryField()[Ki];
        }
    }

    // double *fvcDivPtr = new double[nCells]{0.};
    // - face loop
    // for (int facei = 0; facei < nFaces; facei++)
    // {
    //     fvcDivPtr[l[facei]] += phiPtr[facei] * KfPtr[facei];
    //     fvcDivPtr[u[facei]] -= phiPtr[facei] * KfPtr[facei];
    // }

    // - boundary loop
    // forAll(mesh.boundary(), patchi)
    // {
    //     const labelUList& pFaceCells =
    //         mesh.boundary()[patchi].faceCells();

    //     const fvsPatchField<scalar>& pssf = Kf.boundaryField()[patchi];
    //     const fvsPatchField<scalar>& phisf = phi.boundaryField()[patchi];

    //     // forAll(mesh.boundary()[patchi], facei)
    //     // {
    //     //     fvcDivPtr[pFaceCells[facei]] += pssf[facei] * phisf[facei];
    //     // }
    // }

    time_end = MPI_Wtime();
    Info << "fvcDivTime = " << time_end - time_begin << endl;

    // fvcddt
    time_begin = MPI_Wtime();
    // auto fvcDdtTmp = EulerDdtSchemeFvcDdt(rho, K);
    time_end = MPI_Wtime();
    Info << "fvcDdtTime = " << time_end - time_begin << endl;

    // laplacian
    time_begin = MPI_Wtime();

    // auto fvmLaplacianTmp = gaussLaplacianSchemeFvmLaplacian(alphaEff, he);

    // - get weight
    const scalar* const __restrict__ linearWeightsPtr = linear_weights.primitiveField().begin(); // get linear weight

    // tmp<surfaceInterpolationScheme<scalar>> tinterpGammaScheme_(new linear<scalar>(mesh));
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));

    // tmp<surfaceScalarField> tgamma = tinterpGammaScheme_().interpolate(alphaEff);
    // - interpolation
    tmp<surfaceScalarField> tAlphaf(
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+alphaEff.name()+')',
                alphaEff.instance(),
                alphaEff.db()
            ),
            mesh,
            mesh.Sf().dimensions()*alphaEff.dimensions()
        )
    );
    surfaceScalarField& Alphaf = tAlphaf.ref();
    scalar* alphafPtr = &Alphaf[0];
    const scalar* const __restrict__ alphaPtr = alphaEff.primitiveField().begin();

    for(label f = 0; f < nFaces; ++f){
        alphafPtr[f] = linearWeightsPtr[f] * (alphaPtr[l[f]] - alphaPtr[u[f]]) + alphaPtr[u[f]];
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[Ki];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[Ki];
        fvsPatchScalarField& psf = Alphaf.boundaryFieldRef()[Ki];

        if (alphaEff.boundaryField()[Ki].coupled())
        {
            psf =
                (
                    pLambda*alphaEff.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*alphaEff.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = alphaEff.boundaryField()[Ki];
        }
    }

    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(he)();

    surfaceScalarField gammaMagSf
    (
        Alphaf * mesh.magSf()
    );
    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();


    // forAll(he.boundaryField(), patchi)
    // {
    //     const fvPatchField<scalar>& pvf = he.boundaryField()[patchi];
    //     const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
    //     const fvsPatchScalarField& pDeltaCoeffs =
    //         deltaCoeffs.boundaryField()[patchi];

    //     if (pvf.coupled())
    //     {
    //         fvm.internalCoeffs()[patchi] =
    //             pGamma*pvf.gradientInternalCoeffs(pDeltaCoeffs);
    //         fvm.boundaryCoeffs()[patchi] =
    //         -pGamma*pvf.gradientBoundaryCoeffs(pDeltaCoeffs);
    //     }
    //     else
    //     {
    //         fvm.internalCoeffs()[patchi] = pGamma*pvf.gradientInternalCoeffs();
    //         fvm.boundaryCoeffs()[patchi] = -pGamma*pvf.gradientBoundaryCoeffs();
    //     }
    // }

    time_end = MPI_Wtime();
    Info << "fvmLaplacianTime = " << time_end - time_begin << endl;

    // merge
    time_begin = MPI_Wtime();

    double *fvcDivPtr = new double[nCells]{0.};

    // - face loop
    double var1, var2, var3;
    for(label f = 0; f < nFaces; ++f){
        var1 = - weightsPtr[f] * phiPtr[f];
        var2 = (- weightsPtr[f] + 1.) * phiPtr[f];
        var3 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        lowerPtr[f] += var1 - var3;
        upperPtr[f] += var2 - var3;
        diagPtr[l[f]] += var3 - var1;
        diagPtr[u[f]] += var3 - var1;
        fvcDivPtr[l[f]] += phiPtr[f] * KfPtr[f];
        fvcDivPtr[u[f]] -= phiPtr[f] * KfPtr[f];
    }

    // for(label f = 0; f < nFaces; ++f){
    //     var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
    //     lowerPtr[f] -= var1;
    //     upperPtr[f] -= var1;
    //     diagPtr[l[f]] += var1; //negSumDiag
    //     diagPtr[u[f]] += var1;
    // }

    // - boundary loop
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchField<scalar>& psf = he.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = linear_weights.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs =
            deltaCoeffs.boundaryField()[patchi];

        const fvsPatchField<scalar>& pssf = Kf.boundaryField()[patchi];
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            fvm.internalCoeffs()[patchi] =
                pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs) + patchFlux*psf.valueInternalCoeffs(pw);
            fvm.boundaryCoeffs()[patchi] =
            -pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs) - patchFlux*psf.valueBoundaryCoeffs(pw);
        }
        else
        {
            fvm.internalCoeffs()[patchi] = pGamma*psf.gradientInternalCoeffs() + patchFlux*psf.valueInternalCoeffs(pw);
            fvm.boundaryCoeffs()[patchi] = -pGamma*psf.gradientBoundaryCoeffs() - patchFlux*psf.valueBoundaryCoeffs(pw);
        }
        forAll(mesh.boundary()[patchi], facei)
        {
            fvcDivPtr[pFaceCells[facei]] += pssf[facei] * patchFlux[facei];
        }
    }

    int sourceSum;
    for(label c = 0; c < nCells; ++c){      // cell loop
        // ddt
        diagPtr[c] += rDeltaT * rhoPtr[c] * meshVscPtr[c];
        sourcePtr[c] += rDeltaT * rhoOldTimePtr[c] * heOldTimePtr[c] * meshVscPtr[c];
        // source
        // sourceSum = dpdt[c] - fvcDdtTmp()[c] - fvcDivTmp()[c];
        sourceSum = dpdt[c] 
            - rDeltaT * heOldTimePtr[c] * (rhoPtr[c] - rhoOldTimePtr[c]);
            // - fvcDivTmp()[c];
        sourcePtr[c] += sourceSum * meshVPtr[c];
        sourcePtr[c] -= diffAlphaD[c] * meshVPtr[c];
        sourcePtr[c] -= fvcDivPtr[c];

    }


    // fvScalarMatrix fvm_final
    // (
    //     // fvm + fvmDivTmp - fvmLaplacianTmp 
    //     fvm 
    //     // + fvcDdtTmp 
    //     // + fvcDivTmp
    //     // - dpdt 
    //     // + diffAlphaD
    // );
    time_end = MPI_Wtime();
    Info << "mergeTime = " << time_end - time_begin << endl;
    double total_end = MPI_Wtime();
    Info << "totalTime = " << total_end - total_begin << endl;
    return tfvm;
}

}
