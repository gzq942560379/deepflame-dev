#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

namespace Foam{

tmp<fvScalarMatrix>
GenMatrix_Y(
    const volScalarField& rho,
    volScalarField& Yi,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiUc,
    const volScalarField& rhoD,
    const volScalarField& mut,
    const Switch splitting,
    const scalar Sct,
    CombustionModel<basicThermo>& combustion,
    fv::convectionScheme<scalar>& mvConvection,
    labelList& face_scheduling
){
    assert(splitting == false);

    const fvMesh& mesh = Yi.mesh();
    assert(mesh.moving() == false);

    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();

    // div
    auto& mgcs = dynamic_cast<fv::multivariateGaussConvectionScheme<scalar>&>(mvConvection);
    tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_ = mgcs.interpolationScheme()()(Yi);
    tmp<surfaceScalarField> tweights = tinterpScheme_().weights(Yi);
    const surfaceScalarField& weights = tweights();

    // laplacian
    tmp<surfaceInterpolationScheme<scalar>> tinterpGammaScheme_(new linear<scalar>(mesh));
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
    tmp<volScalarField> gammaScalarVol = rhoD + mut/Sct;

    tmp<surfaceScalarField> tgamma = tinterpGammaScheme_().interpolate(gammaScalarVol);
    const surfaceScalarField& gamma = tgamma.ref();
    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(Yi)();

    const label specieI = combustion.chemistry()->species()[Yi.member()];
    const DimensionedField<scalar, volMesh>& su = combustion.chemistry()->RR(specieI);

    surfaceScalarField gammaMagSf
    (
        gamma * mesh.magSf()
    );

    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            Yi,
            rho.dimensions()*Yi.dimensions()*dimVol/dimTime
        )
    );
    fvScalarMatrix& fvm = tfvm.ref();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    // fvm.diag() = rDeltaT*rho.primitiveField()*mesh.Vsc();
    // fvm.source() = rDeltaT
    //     *rho.oldTime().primitiveField()
    //     *Yi.oldTime().primitiveField()*mesh.Vsc();

    scalar* __restrict__ diagPtr_ddt = fvm.diag().begin();
    scalar* __restrict__ sourcePtr_ddt = fvm.source().begin();
    scalar* __restrict__ lowerPtr_ddt = fvm.lower().begin();
    scalar* __restrict__ upperPtr_ddt = fvm.upper().begin();

    const labelUList& l = fvm.lduAddr().lowerAddr();
    const labelUList& u = fvm.lduAddr().upperAddr();

    // ddt
    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
    const scalar* const __restrict__ YiOldTimePtr = Yi.oldTime().primitiveField().begin();

    // div
    const scalar* const __restrict__ weightsPtr = weights.primitiveField().begin();
    const scalar* const __restrict__ phiPtr = phi.primitiveField().begin();
    const scalar* const __restrict__ phiUcPtr = phiUc.primitiveField().begin();
    
    // laplacian
    // deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();
    // const scalar* const __restrict__ gammaPtr = gamma.primitiveField().begin();
    // const scalar* const __restrict__ meshMagSfPtr = mesh.magSf().primitiveField().begin();

    // combustion->R(Yi)
    // fvm.source() += su.mesh().V()*su.field();
    const scalar* const __restrict__ meshVPtr = mesh.V().begin();
    const scalar* const __restrict__ suFieldVPtr = su.field().begin();


    #pragma omp parallel for
    for(label c = 0; c < nCells; ++c){
        diagPtr_ddt[c] = rDeltaT * rhoPtr[c] * meshVscPtr[c];
        sourcePtr_ddt[c] = rDeltaT * rhoOldTimePtr[c] * YiOldTimePtr[c] * meshVscPtr[c] + meshVPtr[c] * suFieldVPtr[c];
    }

    // fvm_div1.lower() = -weights.primitiveField()*phi.primitiveField();
    // fvm_div1.upper() = fvm_div1.lower() + phi.primitiveField();
    // fvm_laplacian.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    
    #pragma omp parallel for
    for(label f = 0; f < nFaces; ++f){
        lowerPtr_ddt[f] = - weightsPtr[f] * (phiPtr[f] + phiUcPtr[f]) - deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        // upperPtr_ddt[f] = (- weightsPtr[f] + 1.) * (phiPtr[f] + phiUcPtr[f]);
        upperPtr_ddt[f] = (- weightsPtr[f] + 1.) * (phiPtr[f] + phiUcPtr[f]) - deltaCoeffsPtr[f] * gammaMagSfPtr[f];
    }

    // #pragma omp parallel for
    // #pragma clang loop unroll_count(4)
    // #pragma clang loop vectorize(enable)

    // fvm_laplacian.negSumDiag();
    // for (label face=0; face< nFaces; ++face)
    // {
    //     diagPtr_ddt[l[face]] -= lowerPtr_ddt[face];
    //     diagPtr_ddt[u[face]] -= upperPtr_ddt[face];
    // }

    #pragma omp parallel for
    for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label face = face_start; face < face_end; ++face){
            diagPtr_ddt[l[face]] -= lowerPtr_ddt[face];
            diagPtr_ddt[u[face]] -= upperPtr_ddt[face];
        }
    }
    #pragma omp parallel for
    for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for(label face = face_start; face < face_end; ++face){
            diagPtr_ddt[l[face]] -= lowerPtr_ddt[face];
            diagPtr_ddt[u[face]] -= upperPtr_ddt[face];
        }
    }

    forAll(Yi.boundaryField(), patchi)
    {
        const fvPatchField<scalar>& psf = Yi.boundaryField()[patchi];

        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs = deltaCoeffs.boundaryField()[patchi];

        const auto& psfValueInternalCoeffs = psf.valueInternalCoeffs(pw);
        const auto& psfValueBoundaryCoeffs = psf.valueBoundaryCoeffs(pw);


        const fvsPatchScalarField& patchFlux_phi = phi.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux_phiUc = phiUc.boundaryField()[patchi];

        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];

        const scalar* const __restrict__ patchFlux_phiPtr = patchFlux_phi.begin();
        const scalar* const __restrict__ patchFlux_phiUcPtr = patchFlux_phiUc.begin();
        const scalar* const __restrict__ pGammaPtr = pGamma.begin();
        const scalar* const __restrict__ psfValueInternalCoeffsPtr = psfValueInternalCoeffs->begin();
        const scalar* const __restrict__ psfValueBoundaryCoeffsPtr = psfValueBoundaryCoeffs->begin();


        auto& internalCoeffs = fvm.internalCoeffs()[patchi];
        auto& boundaryCoeffs = fvm.boundaryCoeffs()[patchi];

        scalar* __restrict__ internalCoeffsPtr = internalCoeffs.begin();
        scalar* __restrict__ boundaryCoeffsPtr = boundaryCoeffs.begin();
    

        if (psf.coupled())
        {
            const auto& psfGradientInternalCoeffs = psf.gradientInternalCoeffs(pDeltaCoeffs);
            const auto& psfGradientBoundaryCoeffs = psf.gradientBoundaryCoeffs(pDeltaCoeffs);
            const scalar* const __restrict__ psfGradientInternalCoeffsPtr = psfGradientInternalCoeffs->begin();
            const scalar* const __restrict__ psfGradientBoundaryCoeffsPtr = psfGradientBoundaryCoeffs->begin();

            // Info << "psf.coupled()" << endl;

            // fvm.internalCoeffs()[patchi] = (patchFlux_phi + patchFlux_phiUc) * psf.valueInternalCoeffs(pw) - pGamma * psf.gradientInternalCoeffs(pDeltaCoeffs);
            // fvm.boundaryCoeffs()[patchi] = - (patchFlux_phi + patchFlux_phiUc) * psf.valueBoundaryCoeffs(pw) + pGamma * psf.gradientBoundaryCoeffs(pDeltaCoeffs);
            #pragma omp parallel for
            for(label i = 0; i < internalCoeffs.size(); ++i){
                internalCoeffsPtr[i] = (patchFlux_phiPtr[i] + patchFlux_phiUcPtr[i]) * psfValueInternalCoeffsPtr[i] - pGammaPtr[i] * psfGradientInternalCoeffsPtr[i];
                boundaryCoeffsPtr[i] =  - (patchFlux_phiPtr[i] + patchFlux_phiUcPtr[i]) * psfValueBoundaryCoeffsPtr[i] + pGammaPtr[i] * psfGradientBoundaryCoeffsPtr[i];
            }
        }
        else
        {
            // Info << "not psf.coupled()" << endl;
            const auto& psfGradientInternalCoeffs = psf.gradientInternalCoeffs();
            const auto& psfGradientBoundaryCoeffs = psf.gradientBoundaryCoeffs();
            const scalar* const __restrict__ psfGradientInternalCoeffsPtr = psfGradientInternalCoeffs->begin();
            const scalar* const __restrict__ psfGradientBoundaryCoeffsPtr = psfGradientBoundaryCoeffs->begin();
            // fvm.internalCoeffs()[patchi] = (patchFlux_phi + patchFlux_phiUc) * psf.valueInternalCoeffs(pw) - pGamma * psf.gradientInternalCoeffs();
            // fvm.boundaryCoeffs()[patchi] = - (patchFlux_phi + patchFlux_phiUc) * psf.valueBoundaryCoeffs(pw) + pGamma * psf.gradientBoundaryCoeffs();
            #pragma omp parallel for
            for(label i = 0; i < internalCoeffs.size(); ++i){
                internalCoeffsPtr[i] = (patchFlux_phiPtr[i] + patchFlux_phiUcPtr[i]) * psfValueInternalCoeffsPtr[i] - pGammaPtr[i] * psfGradientInternalCoeffsPtr[i];
                boundaryCoeffsPtr[i] =  - (patchFlux_phiPtr[i] + patchFlux_phiUcPtr[i]) * psfValueBoundaryCoeffsPtr[i] + pGammaPtr[i] * psfGradientBoundaryCoeffsPtr[i];
            }
        }
    }


    return tfvm;
}


}

