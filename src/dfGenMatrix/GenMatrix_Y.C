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
    const double *weightsPtr = &weights[0];
    const double* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];

    Info << "nProcessBoundarySurfaces = " << nProcessBoundarySurfaces << endl;
    Info << "nBoundarySurfaces = " << nBoundarySurfaces << endl;
    // allocate memory
    double *boundary_Y = new double[nSpecies * nProcessBoundarySurfaces];
    double *boundary_sf = new double[nProcessBoundarySurfaces * 3];
    double *boundary_mag_sf = new double[nProcessBoundarySurfaces];
    double *boundary_weights = new double[nProcessBoundarySurfaces];
    int *boundary_face_cell = new int[nProcessBoundarySurfaces];
    double *gradY = new double[nSpecies * nCells * 3];
    double *boundary_gradY = new double[nSpecies * nProcessBoundarySurfaces * 3];
    const double *meshVPtr = &mesh.V()[0];
    double ** YiPtr = new double*[nSpecies];
    double *sumYDiffErrorPtr = &sumYDiffError[0][0];
    double *boundary_sumYDiffErrorPtr = new double[nProcessBoundarySurfaces * 3]();
    const double *alphaPtr = &alpha[0];
    double *boundary_alpha = new double[nProcessBoundarySurfaces]();
    double *phiUcPtr = &phiUc[0];
    const double *phiPtr = &phi[0];
    double *boundary_phiUc = new double[nProcessBoundarySurfaces]();
    double *upwindWeightsPtr = &upwindWeights[0];
    double *rhoD = new double[nCells * nSpecies];
    double *boundary_rhoD = new double[nProcessBoundarySurfaces * nSpecies];

    int offset;
    for (int i = 0; i < nSpecies; ++i) {
        YiPtr[i] = &Y[i][0];
        offset = 0;
        forAll(Y[i].boundaryField(), patchi) {
            const fvPatchScalarField& patchYi = Y[i].boundaryField()[patchi];
            const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();
            const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
            const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
            const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
            const fvPatchScalarField& patchAlpha = alpha.boundaryField()[patchi];

            int patchsize = patchYi.size();
            if (patchYi.type() == "processor"
                || patchYi.type() == "processorCyclic") {
                scalarField patchYiInternal =
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchYi).patchInternalField()();
                memcpy(boundary_Y + i * nProcessBoundarySurfaces + offset, &patchYi[0], patchsize * sizeof(double));
                memcpy(boundary_Y + i * nProcessBoundarySurfaces + offset + patchsize, &patchYiInternal[0], patchsize * sizeof(double));

                memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));
                memcpy(boundary_face_cell + offset + patchsize, &pFaceCells[0], patchsize * sizeof(int));

                memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
                memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(double));

                memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
                memcpy(boundary_weights + offset + patchsize, &pWeights[0], patchsize*sizeof(double));

                memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
                memcpy(boundary_mag_sf + offset + patchsize, &pMagSf[0], patchsize*sizeof(double));
                
                memcpy(boundary_alpha + offset, &patchAlpha[0], patchsize * sizeof(double));
                scalarField patchAlphaInternal = 
                        dynamic_cast<const processorFvPatchField<scalar>&>(patchAlpha).patchInternalField()();
                memcpy(boundary_alpha + offset + patchsize, &patchAlphaInternal[0], patchsize * sizeof(double));

                offset += patchsize * 2;
            } else {
                memcpy(boundary_Y + i * nProcessBoundarySurfaces + offset, &patchYi[0], patchsize*sizeof(double));
                memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));
                memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
                memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
                memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
                memcpy(boundary_alpha + offset, &patchAlpha[0], patchsize * sizeof(double));

                offset += patchsize;
            }
        }
    }
    int *patchType = new int[nBoundaryPatches];
    for (int i = 0; i < nBoundaryPatches; ++i) {
        if (Y[0].boundaryField()[i].type() == "processor") {
            patchType[i] = boundaryConditions::processor;
        } else if (Y[0].boundaryField()[i].type() == "zeroGradient") {
            patchType[i] = boundaryConditions::zeroGradient;
        } else {
            Info << "boundary condition not supported" << endl;
            std::exit(-1);
        }
    }

    for (int i = 0; i < nSpecies; ++i) {
        memcpy(rhoD + i * nCells, &chemistry->rhoD(i)[0], nCells * sizeof(double));
        offset = 0;
        forAll(Y[i].boundaryField(), patchi) {
            fvPatchScalarField& patchRhoD = const_cast<fvPatchScalarField&>(chemistry->rhoD(i).boundaryField()[patchi]);
            int patchsize = patchRhoD.size();
            memcpy(boundary_rhoD + i * nProcessBoundarySurfaces + offset, &patchRhoD[0], patchsize * sizeof(double));
            if (patchType[patchi] == boundaryConditions::processor)
                offset += 2 * surfacePerPatch[patchi];
            else
                offset += surfacePerPatch[patchi];
        }
    }

    for (int i = 0; i < nSpecies; ++i) {
        // fvc::grad(Y[i]) inner
        double *Yi = YiPtr[i];
        for (label j = 0; j < nFaces; ++j) {
            double ssf = (weightsPtr[j] * (Yi[own[j]] - Yi[nei[j]]) + Yi[nei[j]]);
            double grad_x = meshSfPtr[3 * j + 0] * ssf;
            double grad_y = meshSfPtr[3 * j + 1] * ssf;
            double grad_z = meshSfPtr[3 * j + 2] * ssf;
            gradY[i * nCells * 3 + own[j] * 3 + 0] += grad_x;
            gradY[i * nCells * 3 + own[j] * 3 + 1] += grad_y;
            gradY[i * nCells * 3 + own[j] * 3 + 2] += grad_z;
            gradY[i * nCells * 3 + nei[j] * 3 + 0] -= grad_x;
            gradY[i * nCells * 3 + nei[j] * 3 + 1] -= grad_y;
            gradY[i * nCells * 3 + nei[j] * 3 + 2] -= grad_z;
        }
        
        // fvc::grad(Y[i]) boundary
        offset = 0;
        for (int j = 0; j < nBoundaryPatches; ++j) {
            if (patchType[j] == boundaryConditions::zeroGradient) {
                for (int k = 0; k < surfacePerPatch[j]; ++k) {
                    int startIndex = offset + k;
                    double bouvf = boundary_Y[i * nProcessBoundarySurfaces + startIndex];
                    double bouSfx = boundary_sf[startIndex * 3 + 0];
                    double bouSfy = boundary_sf[startIndex * 3 + 1];
                    double bouSfz = boundary_sf[startIndex * 3 + 2];
                    double grad_x = bouSfx * bouvf;
                    double grad_y = bouSfy * bouvf;
                    double grad_z = bouSfz * bouvf;

                    gradY[i * nCells * 3 + boundary_face_cell[startIndex] * 3 + 0] += grad_x;
                    gradY[i * nCells * 3 + boundary_face_cell[startIndex] * 3 + 1] += grad_y;
                    gradY[i * nCells * 3 + boundary_face_cell[startIndex] * 3 + 2] += grad_z;
                }
                offset += surfacePerPatch[j];

            } else if (patchType[j] == boundaryConditions::processor) {
                for (int k = 0; k < surfacePerPatch[j]; ++k) {
                    int neighbor_start_index = offset + k;
                    int internal_start_index = offset + surfacePerPatch[j] + k;
                    double bouWeight = boundary_weights[neighbor_start_index];

                    double bouSfx = boundary_sf[neighbor_start_index * 3 + 0];
                    double bouSfy = boundary_sf[neighbor_start_index * 3 + 1];
                    double bouSfz = boundary_sf[neighbor_start_index * 3 + 2];

                    double bouvf = (1 - bouWeight) * boundary_Y[i * nProcessBoundarySurfaces + neighbor_start_index] + 
                            bouWeight * boundary_Y[i * nProcessBoundarySurfaces + internal_start_index];
                    
                    double grad_x = bouSfx * bouvf;
                    double grad_y = bouSfy * bouvf;
                    double grad_z = bouSfz * bouvf;

                    gradY[i * nCells * 3 + boundary_face_cell[neighbor_start_index] * 3 + 0] += grad_x;
                    gradY[i * nCells * 3 + boundary_face_cell[neighbor_start_index] * 3 + 1] += grad_y;
                    gradY[i * nCells * 3 + boundary_face_cell[neighbor_start_index] * 3 + 2] += grad_z;
                }
                offset += 2 * surfacePerPatch[j];
            }
        }

        for (label j = 0; j < nCells; ++j) {
            gradY[i * nCells * 3 + j * 3 + 0] /= meshVPtr[j];
            gradY[i * nCells * 3 + j * 3 + 1] /= meshVPtr[j];
            gradY[i * nCells * 3 + j * 3 + 2] /= meshVPtr[j];
        }

        offset = 0;
        for (int j = 0; j < nBoundaryPatches; ++j) {
            if (patchType[j] == boundaryConditions::zeroGradient) {
                for (int k = 0; k < surfacePerPatch[j]; ++k) {
                    int start_index = offset + k;
                    int cellIndex = boundary_face_cell[start_index];

                    double grad_x = gradY[i * nCells * 3 + cellIndex * 3 + 0];
                    double grad_y = gradY[i * nCells * 3 + cellIndex * 3 + 1];
                    double grad_z = gradY[i * nCells * 3 + cellIndex * 3 + 2];

                    double n_x = boundary_sf[start_index * 3 + 0] / boundary_mag_sf[start_index];
                    double n_y = boundary_sf[start_index * 3 + 1] / boundary_mag_sf[start_index];
                    double n_z = boundary_sf[start_index * 3 + 2] / boundary_mag_sf[start_index];

                    double grad_correction = -(n_x * grad_x + n_y * grad_y + n_z * grad_z);

                    boundary_gradY[i * nProcessBoundarySurfaces * 3 + start_index * 3 + 0] = grad_x + grad_correction * n_x;
                    boundary_gradY[i * nProcessBoundarySurfaces * 3 + start_index * 3 + 1] = grad_y + grad_correction * n_y;
                    boundary_gradY[i * nProcessBoundarySurfaces * 3 + start_index * 3 + 2] = grad_z + grad_correction * n_z;
                }
                offset += surfacePerPatch[j];
            } else if (patchType[j] == boundaryConditions::processor) {
                for (int k = 0; k < surfacePerPatch[j]; ++k) {
                    int neighbor_start_index = offset + k;
                    int internal_start_index = offset + surfacePerPatch[j] + k;
                    int cellIndex = boundary_face_cell[neighbor_start_index];
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

    // // copy gradY to gradResult
    // for (int i = 0; i < nSpecies; ++i) {
    //     volVectorField& gradYi = gradResult[i];
    //     memcpy(&gradYi[0][0], &gradY[i * nCells * 3], nCells * 3 * sizeof(double));
    //     offset = 0;
    //     for (int j = 0; j < nBoundaryPatches; ++j) {
    //         fvPatchVectorField& patchYi = const_cast<fvPatchVectorField&>(gradYi.boundaryField()[j]);
    //         int patchSize = patchYi.size();
    //         memcpy(&patchYi[0][0], &boundary_gradY[i * nProcessBoundarySurfaces * 3 + offset * 3], patchSize * 3 * sizeof(double));
    //         if (patchType[j] == boundaryConditions::processor)
    //             offset += 2 * surfacePerPatch[j];
    //         else
    //             offset += surfacePerPatch[j];
    //     }
    // }
    
    // calculate sumYDiffError
    for (int i = 0; i < nSpecies; ++i) {
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
    for (int j = 0; j < nBoundaryPatches; ++j) {
        fvPatchVectorField& patchSumYi = const_cast<fvPatchVectorField&>(sumYDiffError.boundaryField()[j]);
        int patchSize = patchSumYi.size();
        memcpy(&patchSumYi[0][0], &boundary_sumYDiffErrorPtr[offset * 3], patchSize * 3 * sizeof(double));
        if (patchType[j] == boundaryConditions::processor)
            offset += 2 * surfacePerPatch[j];
        else
            offset += surfacePerPatch[j];
    }

    // calculate phiUc
    // internal
    for (int i = 0; i < nFaces; ++i) {
        double sfx = meshSfPtr[3 * i + 0];
        double sfy = meshSfPtr[3 * i + 1];
        double sfz = meshSfPtr[3 * i + 2];

        int owner = own[i];
        int neighbor = nei[i];

        double w = weightsPtr[i];
        double ssfx = (w * (sumYDiffErrorPtr[3 * owner + 0] - sumYDiffErrorPtr[3 * neighbor + 0]) + sumYDiffErrorPtr[3 * neighbor + 0]);
        double ssfy = (w * (sumYDiffErrorPtr[3 * owner + 1] - sumYDiffErrorPtr[3 * neighbor + 1]) + sumYDiffErrorPtr[3 * neighbor + 1]);
        double ssfz = (w * (sumYDiffErrorPtr[3 * owner + 2] - sumYDiffErrorPtr[3 * neighbor + 2]) + sumYDiffErrorPtr[3 * neighbor + 2]);

        phiUcPtr[i] = sfx * ssfx + sfy * ssfy + sfz * ssfz;
    }
    // boundary
    for (int i = 0; i < nProcessBoundarySurfaces; ++i) {
        double boundary_sfx = boundary_sf[3 * i + 0];
        double boundary_sfy = boundary_sf[3 * i + 1];
        double boundary_sfz = boundary_sf[3 * i + 2];

        double boundary_ssfx = boundary_sumYDiffErrorPtr[3 * i + 0];
        double boundary_ssfy = boundary_sumYDiffErrorPtr[3 * i + 1];
        double boundary_ssfz = boundary_sumYDiffErrorPtr[3 * i + 2];

        boundary_phiUc[i] = boundary_sfx * boundary_ssfx + boundary_sfy * boundary_ssfy + boundary_sfz * boundary_ssfz;
    }

    // copy result to phiUc for test
    offset = 0;
    for (int j = 0; j < nBoundaryPatches; ++j) {
        fvsPatchScalarField& patchPhiUc = const_cast<fvsPatchScalarField&>(phiUc.boundaryField()[j]);
        int patchSize = patchPhiUc.size();
        memcpy(&patchPhiUc[0], &boundary_phiUc[offset], patchSize * sizeof(double));
        if (patchType[j] == boundaryConditions::processor)
            offset += 2 * surfacePerPatch[j];
        else
            offset += surfacePerPatch[j];
    }

    // calculate upwind weight
    for (int j = 0; j < nFaces; ++j) {
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
    double rDeltaT = 1.0/mesh.time().deltaTValue();

    const label specieI = combustion.chemistry()->species()[Yi.member()];
    const DimensionedField<scalar, volMesh>& su = combustion.chemistry()->RR(specieI);

    double* __restrict__ diagPtr = &fvm.diag()[0];
    double* __restrict__ sourcePtr = &fvm.source()[0];
    double* __restrict__ lowerPtr = &fvm.lower()[0];
    double* __restrict__ upperPtr = &fvm.upper()[0];

    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
    const surfaceScalarField& deltaCoeffsTmp = tsnGradScheme_().deltaCoeffs(Yi)();

    // allocate
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const double *weightsPtr = &weights[0];
    const double *rhoPtr = &rho[0];
    const double *meshVPtr = &mesh.V()[0];
    const double *rhoOldTimePtr = &rho.oldTime()[0];
    const double *YiOldTimePtr = &Yi.oldTime()[0];
    const double *phiPtr = &phi[0];
    const double *phiUcPtr = &phiUc[0];
    const double *rhoDPtr = &rhoD[0];
    const double *deltaCoeffsPtr = &mesh.deltaCoeffs()[0];
    const double *magSfPtr = &mesh.magSf()[0];
    const double *suPtr = &su[0];
    double *boundary_phi = new double[nProcessBoundarySurfaces];
    double *boundary_phiUc = new double[nProcessBoundarySurfaces];
    double *internal_coeffs = new double[nProcessBoundarySurfaces]();
    double *boundary_coeffs = new double[nProcessBoundarySurfaces]();
    double *value_internal_coeffs = new double[nProcessBoundarySurfaces];
    double *value_boundary_coeffs = new double[nProcessBoundarySurfaces];
    double *gradient_internal_coeffs = new double[nProcessBoundarySurfaces];
    double *gradient_boundary_coeffs = new double[nProcessBoundarySurfaces];
    double *boundary_weights = new double[nProcessBoundarySurfaces];
    double *boundary_delta_coeffs = new double[nProcessBoundarySurfaces];
    double *boundary_rhoD = new double[nProcessBoundarySurfaces];
    double *boundary_mag_sf = new double[nProcessBoundarySurfaces];
    double *boundary_upwind_weights = new double[nProcessBoundarySurfaces];
    int offset = 0;
    for (int i = 0; i < nBoundaryPatches; ++i) {
        const fvsPatchScalarField& patchPhi = phi.boundaryField()[i];
        const fvsPatchScalarField& patchPhiUc = phiUc.boundaryField()[i];
        const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[i];
        const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[i];
        const fvPatchScalarField& pRhoD = rhoD.boundaryField()[i];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[i];

        int patchsize = patchPhi.size();
        if (patchPhi.type() == "processor") {
            memcpy(boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
            std::fill(boundary_phi + offset + patchsize, boundary_phi + offset + 2 * patchsize, 0.);

            memcpy(boundary_phiUc + offset, &patchPhiUc[0], patchsize * sizeof(double));
            std::fill(boundary_phiUc + offset + patchsize, boundary_phiUc + offset + 2 * patchsize, 0.);
            
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset + patchsize, &pWeights[0], patchsize*sizeof(double));

            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset + patchsize, &pDeltaCoeffs[0], patchsize*sizeof(double));

            memcpy(boundary_rhoD + offset, &pRhoD[0], patchsize * sizeof(double));
            scalarField patchRhoDInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pRhoD).patchInternalField()();
            memcpy(boundary_rhoD + offset + patchsize, &patchRhoDInternal[0], patchsize * sizeof(double));

            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset + patchsize, &pMagSf[0], patchsize*sizeof(double));
            
            offset += 2 * patchsize;
        } else {
            memcpy(boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
            memcpy(boundary_phiUc + offset, &patchPhiUc[0], patchsize * sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_rhoD + offset, &pRhoD[0], patchsize * sizeof(double));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));

            offset += patchsize;
        }
    }
    int *patchType = new int[nBoundaryPatches];
    for (int i = 0; i < nBoundaryPatches; ++i) {
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
    for (int i = 0; i < nProcessBoundarySurfaces; ++i) {
        if (boundary_phi[i] >= 0) {
            boundary_upwind_weights[i] = 1.;
        } else {
            boundary_upwind_weights[i] = 0.;
        }
    }

    // update boundary coeffs
    offset = 0;
    for (int i = 0; i < nBoundaryPatches; ++i) {
        if (patchType[i] == boundaryConditions::zeroGradient) {
            std::fill(value_internal_coeffs + offset, value_internal_coeffs + offset + surfacePerPatch[i], 1.);
            std::fill(value_boundary_coeffs + offset, value_boundary_coeffs + offset + surfacePerPatch[i], 0.);
            std::fill(gradient_internal_coeffs + offset, gradient_internal_coeffs + offset + surfacePerPatch[i], 0.);
            std::fill(gradient_boundary_coeffs + offset, gradient_boundary_coeffs + offset + surfacePerPatch[i], 0.);

            offset += surfacePerPatch[i];
        } else if (patchType[i] == boundaryConditions::processor) {
            for (int j = 0; j < surfacePerPatch[i]; ++j) {
                int start_index = offset + j;
                double bouDeltaCoeffs = boundary_delta_coeffs[start_index];
                double bouWeight = boundary_upwind_weights[start_index];
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
        double w = upwindWeights[i];
        double f = phiPtr[i];

        double lower_value = (-w) * f;
        double upper_value = (1 - w) * f;
        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;

        label owner = own[i];
        label neighbor = nei[i];
        diagPtr[owner] -= lower_value;
        diagPtr[neighbor] -= upper_value;
    }

    // - boundary
    offset = 0;
    for (int i = 0; i < nBoundaryPatches; ++i) {
        for (int j = 0; j < surfacePerPatch[i]; ++j) {
            int start_index = offset + j;
            double boundary_f = boundary_phi[start_index];
            internal_coeffs[start_index] += value_internal_coeffs[start_index] * boundary_f;
            boundary_coeffs[start_index] -= value_boundary_coeffs[start_index] * boundary_f;
        }
        if (patchType[i] == boundaryConditions::processor) offset += 2 * surfacePerPatch[i];
        else offset += surfacePerPatch[i];
    }


    // fvmDiv(phiUc, Yi)
    // - internal
    for (label i = 0; i < nFaces; ++i) {
        double w = upwindWeights[i];
        double f = phiUcPtr[i];

        double lower_value = (-w) * f;
        double upper_value = (1 - w) * f;
        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;

        label owner = own[i];
        label neighbor = nei[i];
        diagPtr[owner] -= lower_value;
        diagPtr[neighbor] -= upper_value;
    }
    // - boundary
    offset = 0;
    for (int i = 0; i < nBoundaryPatches; ++i) {
        for (int j = 0; j < surfacePerPatch[i]; ++j) {
            int start_index = offset + j;
            double boundary_f = boundary_phiUc[start_index];
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
        double w = weightsPtr[i];
        double face_gamma = w * rhoDPtr[owner] + (1 - w) * rhoDPtr[neighbor];

        double upper_value = -1. * face_gamma * magSfPtr[i] * deltaCoeffsPtr[i];
        double lower_value = upper_value;

        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;

        diagPtr[owner] -= lower_value;
        diagPtr[neighbor] -= upper_value;
    }
    // - boundary
    offset = 0;
    for (int i = 0; i < nBoundaryPatches; ++i) {
        for (int j = 0; j < surfacePerPatch[i]; ++j) {
            int start_index = offset + j;
            double boundary_value = boundary_rhoD[start_index] * boundary_mag_sf[start_index];
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
    for (int i = 0; i < nBoundaryPatches; ++i) {
        for (int j = 0; j < surfacePerPatch[i]; ++j) {
            int start_index = offset + j;
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