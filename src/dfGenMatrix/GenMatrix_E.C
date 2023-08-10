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
    const scalar* const __restrict__ heTimePtr = he.primitiveField().begin();

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
    // - get weights TODO:time consuming
    tmp<fv::convectionScheme<scalar>> cs_k = 
        fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div("+phi.name()+','+K.name()+')'));
    fv::gaussConvectionScheme<scalar>& gcs_k = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs_k.ref());

    tmp<surfaceScalarField> tweightsK = gcs_k.interpScheme().weights(K);
    const surfaceScalarField& weightsK = tweightsK();

    // - pointers
    // const scalar* const __restrict__ weightsPtr = linear_weights.primitiveField().begin(); // get linear weight
    const scalar* const __restrict__ weightsKPtr = weightsK.primitiveField().begin(); // get weight

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
    K.oldTime();
    const scalar* const __restrict__ KPtr = K.primitiveField().begin();
    const scalar* const __restrict__ KOldTimePtr = K.oldTime().primitiveField().begin();

    for(label f = 0; f < nFaces; ++f){
        KfPtr[f] = weightsKPtr[f] * (KPtr[l[f]] - KPtr[u[f]]) + KPtr[u[f]];
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = weightsK.boundaryField()[Ki];
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
    // const scalar* const __restrict__ linearWeightsPtr = linear_weights.primitiveField().begin(); // get linear weight

    tmp<surfaceInterpolationScheme<scalar>> tinterpGammaScheme_(new linear<scalar>(mesh));
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));

    tmp<surfaceScalarField> tlaplacianWeight = tinterpGammaScheme_().weights(alphaEff);
    surfaceScalarField laplacianWeight = tlaplacianWeight();
    const scalar* const __restrict__ laplacianWeightsPtr = laplacianWeight.primitiveField().begin();
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
        alphafPtr[f] = laplacianWeightsPtr[f] * (alphaPtr[l[f]] - alphaPtr[u[f]]) + alphaPtr[u[f]];
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

    // fvcDiv(hDiffCorrFlux)
    time_begin = MPI_Wtime();
    tmp<volScalarField> fvcDivHDiffCorrFlux = gaussDivFvcdiv(hDiffCorrFlux);
    // final try
    // - get weights
    /*
    Istream& divIntScheme = mesh.divScheme("div("+hDiffCorrFlux.name()+')');
    word divScheme(divIntScheme);

    tmp<surfaceInterpolationScheme<vector>> tinterpScheme_ = 
        surfaceInterpolationScheme<vector>::New(mesh, divIntScheme);

    tmp<surfaceScalarField> tweightshDiffCorrFlux= tinterpScheme_().weights(hDiffCorrFlux);
    const surfaceScalarField& weightshDiffCorrFlux = tweightshDiffCorrFlux();

    // - interpolation
    const scalar* const __restrict__ weightshDiffCorrFluxPtr = weightshDiffCorrFlux.primitiveField().begin(); // get weight

    tmp<surfaceScalarField> thDiffCorrFluxf(
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+hDiffCorrFlux.name()+')',
                hDiffCorrFlux.instance(),
                hDiffCorrFlux.db()
            ),
            mesh,
            mesh.Sf().dimensions()*hDiffCorrFlux.dimensions()
        )
    );

    surfaceScalarField hDiffCorrFluxf = thDiffCorrFluxf.ref();
    scalar* hDiffCorrFluxfPtr = &hDiffCorrFluxf[0];

    for(label f = 0; f < nFaces; ++f){
        hDiffCorrFluxfPtr[f] = mesh.Sf()[f] & (weightshDiffCorrFluxPtr[f] * (hDiffCorrFlux[l[f]] - hDiffCorrFlux[u[f]]) + hDiffCorrFlux[u[f]]);
    }

    forAll(weightshDiffCorrFlux.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = weightshDiffCorrFlux.boundaryField()[Ki];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[Ki];
        fvsPatchScalarField& psf = hDiffCorrFluxf.boundaryFieldRef()[Ki];

        if (K.boundaryField()[Ki].coupled())
        {
            psf = pSf
              & (
                    pLambda*hDiffCorrFlux.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*hDiffCorrFlux.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = pSf & hDiffCorrFlux.boundaryField()[Ki];
        }
    }*/

    time_end = MPI_Wtime();
    Info << "fvcDivVectorTime = " << time_end - time_begin << endl;

    // merge
    time_begin = MPI_Wtime();

    double *fvcDivPtr = new double[nCells]{0.};
    double *fvcDiv2Ptr = new double[nCells]{0.};

    // - face loop
    double var1, var2, var3;
    for(label f = 0; f < nFaces; ++f){
        var1 = - weightsPtr[f] * phiPtr[f];
        var2 = - weightsPtr[f] * phiPtr[f] + phiPtr[f];
        lowerPtr[f] += var1;
        upperPtr[f] += var2;
        diagPtr[l[f]] -= var1;
        diagPtr[u[f]] -= var2;
        fvcDivPtr[l[f]] += phiPtr[f] * KfPtr[f];
        fvcDivPtr[u[f]] -= phiPtr[f] * KfPtr[f];
        // fvcDiv2Ptr[l[f]] += hDiffCorrFluxf[f];
        // fvcDiv2Ptr[u[f]] -= hDiffCorrFluxf[f];
    }

    double *diagLaplacPtr = new double[nCells]{0.};
    for(label f = 0; f < nFaces; ++f){
        var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        lowerPtr[f] -= var1;
        upperPtr[f] -= var1;
        diagLaplacPtr[l[f]] += var1; //negSumDiag
        diagLaplacPtr[u[f]] += var1;
    }

    // - boundary loop
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchField<scalar>& psf = he.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw_fvm = weights.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        // const fvsPatchScalarField& phDiffCorrFluxf = hDiffCorrFluxf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs =
            deltaCoeffs.boundaryField()[patchi];

        const fvsPatchField<scalar>& pssf = Kf.boundaryField()[patchi];
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            fvm.internalCoeffs()[patchi] =
                -pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs) + patchFlux*psf.valueInternalCoeffs(pw_fvm);
            fvm.boundaryCoeffs()[patchi] =
                pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs) - patchFlux*psf.valueBoundaryCoeffs(pw_fvm);
        }
        else
        {
            fvm.internalCoeffs()[patchi] = - pGamma*psf.gradientInternalCoeffs() + patchFlux*psf.valueInternalCoeffs(pw_fvm);
            fvm.boundaryCoeffs()[patchi] = pGamma*psf.gradientBoundaryCoeffs() - patchFlux*psf.valueBoundaryCoeffs(pw_fvm);
        }
        forAll(mesh.boundary()[patchi], facei)
        {
            fvcDivPtr[pFaceCells[facei]] += pssf[facei] * patchFlux[facei];
            // fvcDiv2Ptr[pFaceCells[facei]] += phDiffCorrFluxf[facei];
        }
    }

    int sourceSum;
    for(label c = 0; c < nCells; ++c){      // cell loop
        // ddt
        diagPtr[c] += rDeltaT * rhoPtr[c] * meshVscPtr[c];
        diagPtr[c] += diagLaplacPtr[c];
        sourcePtr[c] += rDeltaT * rhoOldTimePtr[c] * heOldTimePtr[c] * meshVscPtr[c];
        // source
        // sourceSum = dpdt[c] - fvcDdtTmp()[c] - fvcDivTmp()[c];
        sourcePtr[c] -= rDeltaT * (KPtr[c] * rhoPtr[c] - KOldTimePtr[c] * rhoOldTimePtr[c]) * meshVscPtr[c];
        sourcePtr[c] -= fvcDivPtr[c];
        sourcePtr[c] += dpdt[c] * meshVPtr[c];
            // - fvcDivTmp()[c];
        // sourcePtr[c] += sourceSum * meshVPtr[c];
        sourcePtr[c] -= diffAlphaD[c] * meshVscPtr[c];
        // sourcePtr[c] += fvcDiv2Ptr[c];
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
    return tfvm - fvcDivHDiffCorrFlux;
}

}
