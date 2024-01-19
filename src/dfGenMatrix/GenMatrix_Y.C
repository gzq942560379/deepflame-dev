#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

namespace Foam{

// pre process
// calculate sumYDiffError
// calculate phiUc
// calculate hDiffCorrFlux = 0
// calculate diffAlphaD = 0

void preProcess_Y(
    PtrList<volScalarField>& Y,
    surfaceScalarField& phiUc,
    volVectorField& sumYDiffError,
    // std::vector<volVectorField>& gradResult,
    dfChemistryModel<basicThermo>* chemistry,
    surfaceScalarField& upwindWeights,
    const volScalarField& alpha,
    const surfaceScalarField& phi
)
{
    const fvMesh& mesh = Y[0].mesh(); 
    assert(mesh.moving() == false);

    // basic variables
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const scalar *weightsPtr = &weights[0];
    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];

    Info << "nProcessBoundarySurfaces = " << nProcessBoundarySurfaces << endl;
    Info << "nBoundarySurfaces = " << nBoundarySurfaces << endl;
    Info << "nBoundaryPatches = " << nBoundaryPatches << endl;
    // allocate memory
    scalar *boundary_Y = new scalar[nSpecies * nProcessBoundarySurfaces];
    scalar *boundary_sf = new scalar[nProcessBoundarySurfaces * 3];
    scalar *boundary_mag_sf = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_weights = new scalar[nProcessBoundarySurfaces];
    label *boundary_face_cell = new label[nProcessBoundarySurfaces];
    scalar *gradY = new scalar[nSpecies * nCells * 3];
    scalar *boundary_gradY = new scalar[nSpecies * nProcessBoundarySurfaces * 3];
    const scalar *meshVPtr = &mesh.V()[0];
    scalar ** YiPtr = new scalar*[nSpecies];
    scalar *sumYDiffErrorPtr = &sumYDiffError[0][0];
    scalar *boundary_sumYDiffErrorPtr = new scalar[nProcessBoundarySurfaces * 3]();
    const scalar *alphaPtr = &alpha[0];
    scalar *boundary_alpha = new scalar[nProcessBoundarySurfaces]();
    scalar *phiUcPtr = &phiUc[0];
    const scalar *phiPtr = &phi[0];
    scalar *boundary_phiUc = new scalar[nProcessBoundarySurfaces]();
    scalar *upwindWeightsPtr = &upwindWeights[0];
    scalar *rhoD = new scalar[nCells * nSpecies];
    scalar *boundary_rhoD = new scalar[nProcessBoundarySurfaces * nSpecies];

    label offset;
    for (label i = 0; i < nSpecies; ++i) {
        YiPtr[i] = &Y[i][0];
        offset = 0;
        forAll(Y[i].boundaryField(), patchi) {
            const fvPatchScalarField& patchYi = Y[i].boundaryField()[patchi];
            const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();
            const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
            const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
            const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
            const fvPatchScalarField& patchAlpha = alpha.boundaryField()[patchi];

            label patchsize = patchYi.size();
            if (patchYi.type() == "processor"
                || patchYi.type() == "processorCyclic") {
                scalarField patchYiInternal =
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchYi).patchInternalField()();
                memcpy(boundary_Y + i * nProcessBoundarySurfaces + offset, &patchYi[0], patchsize * sizeof(scalar));
                memcpy(boundary_Y + i * nProcessBoundarySurfaces + offset + patchsize, &patchYiInternal[0], patchsize * sizeof(scalar));

                memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(label));
                memcpy(boundary_face_cell + offset + patchsize, &pFaceCells[0], patchsize * sizeof(label));

                memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(scalar));
                memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(scalar));

                memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(scalar));
                memcpy(boundary_weights + offset + patchsize, &pWeights[0], patchsize*sizeof(scalar));

                memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(scalar));
                memcpy(boundary_mag_sf + offset + patchsize, &pMagSf[0], patchsize*sizeof(scalar));
                
                memcpy(boundary_alpha + offset, &patchAlpha[0], patchsize * sizeof(scalar));
                scalarField patchAlphaInternal = 
                        dynamic_cast<const processorFvPatchField<scalar>&>(patchAlpha).patchInternalField()();
                memcpy(boundary_alpha + offset + patchsize, &patchAlphaInternal[0], patchsize * sizeof(scalar));

                offset += patchsize * 2;
            } else {
                memcpy(boundary_Y + i * nProcessBoundarySurfaces + offset, &patchYi[0], patchsize*sizeof(scalar));
                memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(label));
                memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(scalar));
                memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(scalar));
                memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(scalar));
                memcpy(boundary_alpha + offset, &patchAlpha[0], patchsize * sizeof(scalar));

                offset += patchsize;
            }
        }
    }
    label *patchType = new label[nBoundaryPatches];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nBoundaryPatches; ++i) {
        if (Y[0].boundaryField()[i].type() == "processor") {
            patchType[i] = boundaryConditions::processor;
        } else if (Y[0].boundaryField()[i].type() == "zeroGradient") {
            patchType[i] = boundaryConditions::zeroGradient;
        } else {
            Info << "boundary condition not supported" << endl;
            std::exit(-1);
        }
    }

    for (label i = 0; i < nSpecies; ++i) {
        memcpy(rhoD + i * nCells, &chemistry->rhoD(i)[0], nCells * sizeof(scalar));
        offset = 0;
        forAll(Y[i].boundaryField(), patchi) {
            fvPatchScalarField& patchRhoD = const_cast<fvPatchScalarField&>(chemistry->rhoD(i).boundaryField()[patchi]);
            label patchsize = patchRhoD.size();
            memcpy(boundary_rhoD + i * nProcessBoundarySurfaces + offset, &patchRhoD[0], patchsize * sizeof(scalar));
            if (patchType[patchi] == boundaryConditions::processor)
                offset += 2 * surfacePerPatch[patchi];
            else
                offset += surfacePerPatch[patchi];
        }
    }

    for (label i = 0; i < nSpecies; ++i) {
        // fvc::grad(Y[i]) inner
        scalar *Yi = YiPtr[i];
        scalar *gradY_Species = &gradY[i * nCells * 3];
        
        for (label j = 0; j < nFaces; ++j) {
            label owner = own[j];
            label neighbor = nei[j];
            scalar ssf = (weightsPtr[j] * (Yi[owner] - Yi[neighbor]) + Yi[neighbor]);
            scalar grad_x = meshSfPtr[3 * j + 0] * ssf;
            scalar grad_y = meshSfPtr[3 * j + 1] * ssf;
            scalar grad_z = meshSfPtr[3 * j + 2] * ssf;
            gradY_Species[owner * 3 + 0] += grad_x;
            gradY_Species[owner * 3 + 1] += grad_y;
            gradY_Species[owner * 3 + 2] += grad_z;
            gradY_Species[neighbor * 3 + 0] -= grad_x;
            gradY_Species[neighbor * 3 + 1] -= grad_y;
            gradY_Species[neighbor * 3 + 2] -= grad_z;
        }
        
        // fvc::grad(Y[i]) boundary

        {
            offset = 0;
            for (label j = 0; j < nBoundaryPatches; ++j) {
                if (patchType[j] == boundaryConditions::zeroGradient) {
                    for (label k = 0; k < surfacePerPatch[j]; ++k) {
                        label startIndex = offset + k;
                        scalar bouvf = boundary_Y[i * nProcessBoundarySurfaces + startIndex];
                        scalar bouSfx = boundary_sf[startIndex * 3 + 0];
                        scalar bouSfy = boundary_sf[startIndex * 3 + 1];
                        scalar bouSfz = boundary_sf[startIndex * 3 + 2];
                        scalar grad_x = bouSfx * bouvf;
                        scalar grad_y = bouSfy * bouvf;
                        scalar grad_z = bouSfz * bouvf;

                        gradY[i * nCells * 3 + boundary_face_cell[startIndex] * 3 + 0] += grad_x;
                        gradY[i * nCells * 3 + boundary_face_cell[startIndex] * 3 + 1] += grad_y;
                        gradY[i * nCells * 3 + boundary_face_cell[startIndex] * 3 + 2] += grad_z;
                    }
                    offset += surfacePerPatch[j];

                } else if (patchType[j] == boundaryConditions::processor) {
                    for (label k = 0; k < surfacePerPatch[j]; ++k) {
                        label neighbor_start_index = offset + k;
                        label internal_start_index = offset + surfacePerPatch[j] + k;
                        scalar bouWeight = boundary_weights[neighbor_start_index];

                        scalar bouSfx = boundary_sf[neighbor_start_index * 3 + 0];
                        scalar bouSfy = boundary_sf[neighbor_start_index * 3 + 1];
                        scalar bouSfz = boundary_sf[neighbor_start_index * 3 + 2];

                        scalar bouvf = (1 - bouWeight) * boundary_Y[i * nProcessBoundarySurfaces + neighbor_start_index] + 
                                bouWeight * boundary_Y[i * nProcessBoundarySurfaces + internal_start_index];
                        
                        scalar grad_x = bouSfx * bouvf;
                        scalar grad_y = bouSfy * bouvf;
                        scalar grad_z = bouSfz * bouvf;

                        gradY[i * nCells * 3 + boundary_face_cell[neighbor_start_index] * 3 + 0] += grad_x;
                        gradY[i * nCells * 3 + boundary_face_cell[neighbor_start_index] * 3 + 1] += grad_y;
                        gradY[i * nCells * 3 + boundary_face_cell[neighbor_start_index] * 3 + 2] += grad_z;
                    }
                    offset += 2 * surfacePerPatch[j];
                }
            }
        }

        for (label j = 0; j < nCells; ++j) {
            gradY[i * nCells * 3 + j * 3 + 0] /= meshVPtr[j];
            gradY[i * nCells * 3 + j * 3 + 1] /= meshVPtr[j];
            gradY[i * nCells * 3 + j * 3 + 2] /= meshVPtr[j];
        }

        {
            offset = 0;
            for (label j = 0; j < nBoundaryPatches; ++j) {
                if (patchType[j] == boundaryConditions::zeroGradient) {
                    for (label k = 0; k < surfacePerPatch[j]; ++k) {
                        label start_index = offset + k;
                        label cellIndex = boundary_face_cell[start_index];

                        scalar grad_x = gradY[i * nCells * 3 + cellIndex * 3 + 0];
                        scalar grad_y = gradY[i * nCells * 3 + cellIndex * 3 + 1];
                        scalar grad_z = gradY[i * nCells * 3 + cellIndex * 3 + 2];

                        scalar n_x = boundary_sf[start_index * 3 + 0] / boundary_mag_sf[start_index];
                        scalar n_y = boundary_sf[start_index * 3 + 1] / boundary_mag_sf[start_index];
                        scalar n_z = boundary_sf[start_index * 3 + 2] / boundary_mag_sf[start_index];

                        scalar grad_correction = -(n_x * grad_x + n_y * grad_y + n_z * grad_z);

                        boundary_gradY[i * nProcessBoundarySurfaces * 3 + start_index * 3 + 0] = grad_x + grad_correction * n_x;
                        boundary_gradY[i * nProcessBoundarySurfaces * 3 + start_index * 3 + 1] = grad_y + grad_correction * n_y;
                        boundary_gradY[i * nProcessBoundarySurfaces * 3 + start_index * 3 + 2] = grad_z + grad_correction * n_z;
                    }
                    offset += surfacePerPatch[j];
                } else if (patchType[j] == boundaryConditions::processor) {
                    Info << "surfacePerPatch[" << j << "] : " << surfacePerPatch[j] << endl;
                    for (label k = 0; k < surfacePerPatch[j]; ++k) {
                        label neighbor_start_index = offset + k;
                        label internal_start_index = offset + surfacePerPatch[j] + k;
                        label cellIndex = boundary_face_cell[neighbor_start_index];
                        // correct_internal_boundary_field_vector
                        boundary_gradY[i * nProcessBoundarySurfaces * 3 + internal_start_index * 3 + 0] = gradY[i * nCells * 3 + cellIndex * 3 + 0];
                        boundary_gradY[i * nProcessBoundarySurfaces * 3 + internal_start_index * 3 + 1] = gradY[i * nCells * 3 + cellIndex * 3 + 1];
                        boundary_gradY[i * nProcessBoundarySurfaces * 3 + internal_start_index * 3 + 2] = gradY[i * nCells * 3 + cellIndex * 3 + 2];
                        // update processor boundary
                        MPI_Send(&boundary_gradY[i * nProcessBoundarySurfaces * 3 + internal_start_index * 3], 3, MPI_DOUBLE, neighbProcNo[j], 0, MPI_COMM_WORLD);
                        MPI_Recv(&boundary_gradY[i * nProcessBoundarySurfaces * 3 + neighbor_start_index * 3], 3, MPI_DOUBLE, neighbProcNo[j], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    offset += 2 * surfacePerPatch[j];
                }
            }
        }
    }


    // // copy gradY to gradResult
    // for (label i = 0; i < nSpecies; ++i) {
    //     volVectorField& gradYi = gradResult[i];
    //     memcpy(&gradYi[0][0], &gradY[i * nCells * 3], nCells * 3 * sizeof(scalar));
    //     offset = 0;
    //     for (label j = 0; j < nBoundaryPatches; ++j) {
    //         fvPatchVectorField& patchYi = const_cast<fvPatchVectorField&>(gradYi.boundaryField()[j]);
    //         label patchSize = patchYi.size();
    //         memcpy(&patchYi[0][0], &boundary_gradY[i * nProcessBoundarySurfaces * 3 + offset * 3], patchSize * 3 * sizeof(scalar));
    //         if (patchType[j] == boundaryConditions::processor)
    //             offset += 2 * surfacePerPatch[j];
    //         else
    //             offset += surfacePerPatch[j];
    //     }
    // }
    
    // calculate sumYDiffError
    for (label i = 0; i < nSpecies; ++i) {
        for (label j = 0; j < nCells; ++j) {
            sumYDiffErrorPtr[3 * j + 0] += gradY[i * nCells * 3 + j * 3 + 0] * rhoD[i * nCells + j];
            sumYDiffErrorPtr[3 * j + 1] += gradY[i * nCells * 3 + j * 3 + 1] * rhoD[i * nCells + j];
            sumYDiffErrorPtr[3 * j + 2] += gradY[i * nCells * 3 + j * 3 + 2] * rhoD[i * nCells + j];
        }
        for (label j = 0; j < nProcessBoundarySurfaces; ++j) {
            boundary_sumYDiffErrorPtr[3 * j + 0] += boundary_gradY[i * nProcessBoundarySurfaces * 3 + j * 3 + 0] * boundary_alpha[j];
            boundary_sumYDiffErrorPtr[3 * j + 1] += boundary_gradY[i * nProcessBoundarySurfaces * 3 + j * 3 + 1] * boundary_alpha[j];
            boundary_sumYDiffErrorPtr[3 * j + 2] += boundary_gradY[i * nProcessBoundarySurfaces * 3 + j * 3 + 2] * boundary_alpha[j];
        }
    }

    // copy result to sumYDiffError
    offset = 0;
    for (label j = 0; j < nBoundaryPatches; ++j) {
        fvPatchVectorField& patchSumYi = const_cast<fvPatchVectorField&>(sumYDiffError.boundaryField()[j]);
        label patchSize = patchSumYi.size();
        memcpy(&patchSumYi[0][0], &boundary_sumYDiffErrorPtr[offset * 3], patchSize * 3 * sizeof(scalar));
        if (patchType[j] == boundaryConditions::processor)
            offset += 2 * surfacePerPatch[j];
        else
            offset += surfacePerPatch[j];
    }

    // calculate phiUc
    // internal
    for (label i = 0; i < nFaces; ++i) {
        scalar sfx = meshSfPtr[3 * i + 0];
        scalar sfy = meshSfPtr[3 * i + 1];
        scalar sfz = meshSfPtr[3 * i + 2];

        label owner = own[i];
        label neighbor = nei[i];

        scalar w = weightsPtr[i];
        scalar ssfx = (w * (sumYDiffErrorPtr[3 * owner + 0] - sumYDiffErrorPtr[3 * neighbor + 0]) + sumYDiffErrorPtr[3 * neighbor + 0]);
        scalar ssfy = (w * (sumYDiffErrorPtr[3 * owner + 1] - sumYDiffErrorPtr[3 * neighbor + 1]) + sumYDiffErrorPtr[3 * neighbor + 1]);
        scalar ssfz = (w * (sumYDiffErrorPtr[3 * owner + 2] - sumYDiffErrorPtr[3 * neighbor + 2]) + sumYDiffErrorPtr[3 * neighbor + 2]);

        phiUcPtr[i] = sfx * ssfx + sfy * ssfy + sfz * ssfz;
    }
    // boundary
    for (label i = 0; i < nProcessBoundarySurfaces; ++i) {
        scalar boundary_sfx = boundary_sf[3 * i + 0];
        scalar boundary_sfy = boundary_sf[3 * i + 1];
        scalar boundary_sfz = boundary_sf[3 * i + 2];

        scalar boundary_ssfx = boundary_sumYDiffErrorPtr[3 * i + 0];
        scalar boundary_ssfy = boundary_sumYDiffErrorPtr[3 * i + 1];
        scalar boundary_ssfz = boundary_sumYDiffErrorPtr[3 * i + 2];

        boundary_phiUc[i] = boundary_sfx * boundary_ssfx + boundary_sfy * boundary_ssfy + boundary_sfz * boundary_ssfz;
    }

    // copy result to phiUc for test
    offset = 0;
    for (label j = 0; j < nBoundaryPatches; ++j) {
        fvsPatchScalarField& patchPhiUc = const_cast<fvsPatchScalarField&>(phiUc.boundaryField()[j]);
        label patchSize = patchPhiUc.size();
        memcpy(&patchPhiUc[0], &boundary_phiUc[offset], patchSize * sizeof(scalar));
        if (patchType[j] == boundaryConditions::processor)
            offset += 2 * surfacePerPatch[j];
        else
            offset += surfacePerPatch[j];
    }

    // calculate upwind weight
    for (label j = 0; j < nFaces; ++j) {
        if (phiPtr[j] >= 0) {
            upwindWeightsPtr[j] = 1.;
        } else {
            upwindWeightsPtr[j] = 0.;
        }
    }

    // free ptrs
    delete[] boundary_Y;
    delete[] boundary_sf;
    delete[] boundary_mag_sf;
    delete[] boundary_weights;
    delete[] boundary_face_cell;
    delete[] gradY;
    delete[] boundary_gradY;
    delete[] YiPtr;
    delete[] boundary_sumYDiffErrorPtr;
    delete[] boundary_alpha;
    delete[] boundary_phiUc;
    delete[] patchType;
    delete[] rhoD;
    delete[] boundary_rhoD;
}

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
    const surfaceScalarField& upwindWeights
){
    assert(splitting == false);

    const fvMesh& mesh = Yi.mesh();
    assert(mesh.moving() == false);

    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];

    // construct fvMatrix
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

    const label specieI = combustion.chemistry()->species()[Yi.member()];
    const DimensionedField<scalar, volMesh>& su = combustion.chemistry()->RR(specieI);

    scalar* __restrict__ diagPtr = &fvm.diag()[0];
    scalar* __restrict__ sourcePtr = &fvm.source()[0];
    scalar* __restrict__ lowerPtr = &fvm.lower()[0];
    scalar* __restrict__ upperPtr = &fvm.upper()[0];

    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
    const surfaceScalarField& deltaCoeffsTmp = tsnGradScheme_().deltaCoeffs(Yi)();

    // allocate
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const scalar *weightsPtr = &weights[0];
    const scalar *rhoPtr = &rho[0];
    const scalar *meshVPtr = &mesh.V()[0];
    const scalar *rhoOldTimePtr = &rho.oldTime()[0];
    const scalar *YiOldTimePtr = &Yi.oldTime()[0];
    const scalar *phiPtr = &phi[0];
    const scalar *phiUcPtr = &phiUc[0];
    const scalar *rhoDPtr = &rhoD[0];
    const scalar *deltaCoeffsPtr = &mesh.deltaCoeffs()[0];
    const scalar *magSfPtr = &mesh.magSf()[0];
    const scalar *suPtr = &su[0];
    scalar *boundary_phi = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_phiUc = new scalar[nProcessBoundarySurfaces];
    scalar *internal_coeffs = new scalar[nProcessBoundarySurfaces]();
    scalar *boundary_coeffs = new scalar[nProcessBoundarySurfaces]();
    scalar *value_internal_coeffs = new scalar[nProcessBoundarySurfaces];
    scalar *value_boundary_coeffs = new scalar[nProcessBoundarySurfaces];
    scalar *gradient_internal_coeffs = new scalar[nProcessBoundarySurfaces];
    scalar *gradient_boundary_coeffs = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_weights = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_delta_coeffs = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_rhoD = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_mag_sf = new scalar[nProcessBoundarySurfaces];
    scalar *boundary_upwind_weights = new scalar[nProcessBoundarySurfaces];
    label offset = 0;
    for (label i = 0; i < nBoundaryPatches; ++i) {
        const fvsPatchScalarField& patchPhi = phi.boundaryField()[i];
        const fvsPatchScalarField& patchPhiUc = phiUc.boundaryField()[i];
        const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[i];
        const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[i];
        const fvPatchScalarField& pRhoD = rhoD.boundaryField()[i];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[i];

        label patchsize = patchPhi.size();
        if (patchPhi.type() == "processor") {
            memcpy(boundary_phi + offset, &patchPhi[0], patchsize * sizeof(scalar));
            std::fill(boundary_phi + offset + patchsize, boundary_phi + offset + 2 * patchsize, 0.);

            memcpy(boundary_phiUc + offset, &patchPhiUc[0], patchsize * sizeof(scalar));
            std::fill(boundary_phiUc + offset + patchsize, boundary_phiUc + offset + 2 * patchsize, 0.);
            
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(scalar));
            memcpy(boundary_weights + offset + patchsize, &pWeights[0], patchsize*sizeof(scalar));

            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(scalar));
            memcpy(boundary_delta_coeffs + offset + patchsize, &pDeltaCoeffs[0], patchsize*sizeof(scalar));

            memcpy(boundary_rhoD + offset, &pRhoD[0], patchsize * sizeof(scalar));
            scalarField patchRhoDInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pRhoD).patchInternalField()();
            memcpy(boundary_rhoD + offset + patchsize, &patchRhoDInternal[0], patchsize * sizeof(scalar));

            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(scalar));
            memcpy(boundary_mag_sf + offset + patchsize, &pMagSf[0], patchsize*sizeof(scalar));
            
            offset += 2 * patchsize;
        } else {
            memcpy(boundary_phi + offset, &patchPhi[0], patchsize * sizeof(scalar));
            memcpy(boundary_phiUc + offset, &patchPhiUc[0], patchsize * sizeof(scalar));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(scalar));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(scalar));
            memcpy(boundary_rhoD + offset, &pRhoD[0], patchsize * sizeof(scalar));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(scalar));

            offset += patchsize;
        }
    }
    label *patchType = new label[nBoundaryPatches];
    for (label i = 0; i < nBoundaryPatches; ++i) {
        if (Yi.boundaryField()[i].type() == "processor") {
            patchType[i] = boundaryConditions::processor;
        } else if (Yi.boundaryField()[i].type() == "zeroGradient") {
            patchType[i] = boundaryConditions::zeroGradient;
        } else {
            Info << "boundary condition not supported" << endl;
            std::exit(-1);
        }
    }

    offset = 0;
    for (label i = 0; i < nProcessBoundarySurfaces; ++i) {
        if (boundary_phi[i] >= 0) {
            boundary_upwind_weights[i] = 1.;
        } else {
            boundary_upwind_weights[i] = 0.;
        }
    }

    // update boundary coeffs
    offset = 0;
    for (label i = 0; i < nBoundaryPatches; ++i) {
        if (patchType[i] == boundaryConditions::zeroGradient) {
            std::fill(value_internal_coeffs + offset, value_internal_coeffs + offset + surfacePerPatch[i], 1.);
            std::fill(value_boundary_coeffs + offset, value_boundary_coeffs + offset + surfacePerPatch[i], 0.);
            std::fill(gradient_internal_coeffs + offset, gradient_internal_coeffs + offset + surfacePerPatch[i], 0.);
            std::fill(gradient_boundary_coeffs + offset, gradient_boundary_coeffs + offset + surfacePerPatch[i], 0.);

            offset += surfacePerPatch[i];
        } else if (patchType[i] == boundaryConditions::processor) {
            for (label j = 0; j < surfacePerPatch[i]; ++j) {
                label start_index = offset + j;
                scalar bouDeltaCoeffs = boundary_delta_coeffs[start_index];
                scalar bouWeight = boundary_upwind_weights[start_index];
                value_internal_coeffs[start_index] = bouWeight;
                value_boundary_coeffs[start_index] = 1 - bouWeight;
                gradient_internal_coeffs[start_index] = -1 * bouDeltaCoeffs;
                gradient_boundary_coeffs[start_index] = bouDeltaCoeffs;
            }
            offset += 2 * surfacePerPatch[i];
        }
    }

    // ddt
    for (label i = 0; i < nCells; ++i) {
        diagPtr[i] += rDeltaT * rhoPtr[i] * meshVPtr[i];
        sourcePtr[i] += rDeltaT * rhoOldTimePtr[i] * YiOldTimePtr[i] * meshVPtr[i];
    }

    // fvmDiv(phi, Yi)
    // - internal
    for (label i = 0; i < nFaces; ++i) {
        scalar w = upwindWeights[i];
        scalar f = phiPtr[i];

        scalar lower_value = (-w) * f;
        scalar upper_value = (1 - w) * f;
        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;

        label owner = own[i];
        label neighbor = nei[i];
        diagPtr[owner] -= lower_value;
        diagPtr[neighbor] -= upper_value;
    }

    // - boundary
    offset = 0;
    for (label i = 0; i < nBoundaryPatches; ++i) {
        for (label j = 0; j < surfacePerPatch[i]; ++j) {
            label start_index = offset + j;
            scalar boundary_f = boundary_phi[start_index];
            internal_coeffs[start_index] += value_internal_coeffs[start_index] * boundary_f;
            boundary_coeffs[start_index] -= value_boundary_coeffs[start_index] * boundary_f;
        }
        if (patchType[i] == boundaryConditions::processor) offset += 2 * surfacePerPatch[i];
        else offset += surfacePerPatch[i];
    }


    // fvmDiv(phiUc, Yi)
    // - internal
    for (label i = 0; i < nFaces; ++i) {
        scalar w = upwindWeights[i];
        scalar f = phiUcPtr[i];

        scalar lower_value = (-w) * f;
        scalar upper_value = (1 - w) * f;
        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;

        label owner = own[i];
        label neighbor = nei[i];
        diagPtr[owner] -= lower_value;
        diagPtr[neighbor] -= upper_value;
    }
    // - boundary
    offset = 0;
    for (label i = 0; i < nBoundaryPatches; ++i) {
        for (label j = 0; j < surfacePerPatch[i]; ++j) {
            label start_index = offset + j;
            scalar boundary_f = boundary_phiUc[start_index];
            internal_coeffs[start_index] += value_internal_coeffs[start_index] * boundary_f;
            boundary_coeffs[start_index] -= value_boundary_coeffs[start_index] * boundary_f;
        }
        if (patchType[i] == boundaryConditions::processor) offset += 2 * surfacePerPatch[i];
        else offset += surfacePerPatch[i];
    }

    // fvm::laplacian(DEff(), Yi)
    // - internal
    for (label i = 0; i < nFaces; ++i) {
        label owner = own[i];
        label neighbor = nei[i];
        scalar w = weightsPtr[i];
        scalar face_gamma = w * rhoDPtr[owner] + (1 - w) * rhoDPtr[neighbor];

        scalar upper_value = -1. * face_gamma * magSfPtr[i] * deltaCoeffsPtr[i];
        scalar lower_value = upper_value;

        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;

        diagPtr[owner] -= lower_value;
        diagPtr[neighbor] -= upper_value;
    }
    // - boundary
    offset = 0;
    for (label i = 0; i < nBoundaryPatches; ++i) {
        for (label j = 0; j < surfacePerPatch[i]; ++j) {
            label start_index = offset + j;
            scalar boundary_value = boundary_rhoD[start_index] * boundary_mag_sf[start_index];
            internal_coeffs[start_index] -= boundary_value * gradient_internal_coeffs[start_index];
            boundary_coeffs[start_index] += boundary_value * gradient_boundary_coeffs[start_index];
        }
        if (patchType[i] == boundaryConditions::processor) offset += 2 * surfacePerPatch[i];
        else offset += surfacePerPatch[i];
    }

    // add source 
    for (label i = 0; i < nCells; ++i) {
        sourcePtr[i] += meshVPtr[i] * suPtr[i];
    }

    // cpoy result to fvm
    offset = 0;
    for (label i = 0; i < nBoundaryPatches; ++i) {
        for (label j = 0; j < surfacePerPatch[i]; ++j) {
            label start_index = offset + j;
            fvm.internalCoeffs()[i][j] = internal_coeffs[start_index];
            fvm.boundaryCoeffs()[i][j] = boundary_coeffs[start_index];
        }
        if (patchType[i] == boundaryConditions::processor) offset += 2 * surfacePerPatch[i];
        else offset += surfacePerPatch[i];
    }

    // free ptrs
    delete[] boundary_phi;
    delete[] boundary_phiUc;
    delete[] internal_coeffs;
    delete[] boundary_coeffs;
    delete[] value_internal_coeffs;
    delete[] value_boundary_coeffs;
    delete[] gradient_internal_coeffs;
    delete[] gradient_boundary_coeffs;
    delete[] boundary_weights;
    delete[] boundary_delta_coeffs;
    delete[] boundary_rhoD;
    delete[] boundary_mag_sf;
    delete[] boundary_upwind_weights;
    delete[] patchType;

    return tfvm;
}

}