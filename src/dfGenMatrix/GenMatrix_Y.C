#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"
#include "clockTime.H"

#define _OPENMP

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
    clockTime clock;
    const fvMesh& mesh = Y[0].mesh(); 
    assert(mesh.moving() == false);

    const MeshSchedule& meshSchedule = MeshSchedule::getMeshSchedule();
    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();

    // basic variables
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const scalar *weightsPtr = &weights[0];
    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];

    Info << "nCells = " << nCells << endl;
    Info << "nFaces = " << nFaces << endl;
    Info << "nPatches = " << nPatches << endl;
    // allocate memory
    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;

    Info << "preProcess_Y init : " << clock.timeIncrement() << endl;

    constScalarPtr* boundaryYi = new constScalarPtr[nSpecies * nPatches];
    scalarPtr* boundaryYiInternal = new scalarPtr[nSpecies * nPatches];
    constLabelPtr* faceCells = new constLabelPtr[nPatches];
    constScalarPtr* boundarySf = new constScalarPtr[nPatches];
    constScalarPtr* boundaryMagSf = new constScalarPtr[nPatches];
    constScalarPtr* boundaryWeights = new constScalarPtr[nPatches];
    constScalarPtr* boundaryAlpha = new constScalarPtr[nPatches];
    scalarPtr* boundaryAlphaInternal = new scalarPtr[nPatches];

    constScalarPtr* YiPtr = new constScalarPtr[nSpecies];
    const scalar* meshVPtr = mesh.V().begin();
    const scalar *phiPtr = phi.begin();
    scalar *upwindWeightsPtr = upwindWeights.begin();

    scalar *gradY = new scalar[nSpecies * nCells * 3];
    scalarPtr* boundaryGradY = new scalarPtr[nSpecies * nPatches];
    scalarPtr* boundaryGradYInternal = new scalarPtr[nSpecies * nPatches];
    
    scalar* sumYDiffErrorPtr = (scalar*)sumYDiffError.begin(); // nCell x 3
    scalarPtr* boundarySumYDiffErrorPtr = new scalarPtr[nPatches];

    scalar *phiUcPtr = phiUc.begin();
    scalarPtr* boundaryPhiUcPtr = new scalarPtr[nPatches];

    constScalarPtr* rhoD = new constScalarPtr[nSpecies];

    for (label i = 0; i < nSpecies; ++i) {
        rhoD[i] = chemistry->rhoD(i).begin(); // nCells
    }
    
    Info << "preProcess_Y alloc : " << clock.timeIncrement() << endl;

    for(label patchi = 0; patchi < nPatches; ++patchi) {
        label patchSize = patchSizes[patchi];
        faceCells[patchi] = mesh.boundary()[patchi].faceCells().begin(); // patchSize
        boundarySf[patchi] = (scalar*) mesh.Sf().boundaryField()[patchi].begin(); // patchSize * 3
        boundaryMagSf[patchi] = mesh.magSf().boundaryField()[patchi].begin(); // patchSize
        boundaryWeights[patchi] = mesh.surfaceInterpolation::weights().boundaryField()[patchi].begin(); // patchSize
        const fvPatchScalarField& patchAlpha = alpha.boundaryField()[patchi];
        boundaryAlpha[patchi] = patchAlpha.begin(); // patchSize

        fvPatchVectorField& patchSumYi = const_cast<fvPatchVectorField&>(sumYDiffError.boundaryField()[patchi]);
        boundarySumYDiffErrorPtr[patchi] = (scalar*)patchSumYi.begin(); // patchSize * 3

        fvsPatchScalarField& patchPhiUc = const_cast<fvsPatchScalarField&>(phiUc.boundaryField()[patchi]);
        boundaryPhiUcPtr[patchi] = patchPhiUc.begin(); // patchSize

        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            tmp<scalarField> tPatchAlphaInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchAlpha).patchInternalField();
            const scalarField& patchAlphaInternal = tPatchAlphaInternal();
            boundaryAlphaInternal[patchi] = new scalar[patchSize];
            
            memcpy(boundaryAlphaInternal[patchi], patchAlphaInternal.begin(), patchSize * sizeof(scalar));
        
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            boundaryAlphaInternal[patchi] = nullptr;
        }else{
            Info << "patch type not supported" << endl;
            std::exit(-1);
        }
    }

    Info << "preProcess_Y alloc copy 1 : " << clock.timeIncrement() << endl;

    for (label i = 0; i < nSpecies; ++i) {
        YiPtr[i] = (const scalar*)Y[i].begin();
        for(label patchi = 0; patchi < nPatches; ++patchi) {
            const fvPatchScalarField& patchYi = Y[i].boundaryField()[patchi];
            label patchSize = patchSizes[patchi];
            boundaryGradY[i * nPatches + patchi] = new scalar[patchSize * 3];
            if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
                // TODO
                tmp<scalarField> tPatchYiInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchYi).patchInternalField();
                const scalarField& patchYiInternal = tPatchYiInternal();
                boundaryYi[i * nPatches + patchi] = patchYi.begin();
                boundaryYiInternal[i * nPatches + patchi] = new scalar[patchSize];
                memcpy(boundaryYiInternal[i * nPatches + patchi], patchYiInternal.begin(), patchSize * sizeof(scalar));
                boundaryGradYInternal[i * nPatches + patchi] = new scalar[patchSize * 3];
            } else {
                boundaryYi[i * nPatches + patchi] = patchYi.begin();
                boundaryYiInternal[i * nPatches + patchi] = nullptr;
                boundaryGradYInternal[i * nPatches + patchi] = nullptr;
            }
        }
    }

    Info << "preProcess_Y alloc copy 2 : " << clock.timeIncrement() << endl;

    double gardY_init_time = 0;
    double gardY_face_time = 0;
    double gardY_boundary_time = 0;
    double gardY_div_time = 0;
    double boundaryGardY_time = 0;
    double boundaryGardY_comm_time = 0;

    for (label i = 0; i < nSpecies; ++i) {
        const scalar *Yi = YiPtr[i];
        scalar *gradY_Species = &gradY[i * nCells * 3];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label c = 0; c < nCells; ++c){
            gradY_Species[c * 3 + 0] = 0.;
            gradY_Species[c * 3 + 1] = 0.;
            gradY_Species[c * 3 + 2] = 0.;
        }

        gardY_init_time += clock.timeIncrement();

#ifdef OPT_OMP_ATOMIC
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label j = 0; j < nFaces; ++j) {
            label owner = own[j];
            label neighbor = nei[j];
            scalar ssf = (weightsPtr[j] * (Yi[owner] - Yi[neighbor]) + Yi[neighbor]);
            scalar grad_x = meshSfPtr[3 * j + 0] * ssf;
            scalar grad_y = meshSfPtr[3 * j + 1] * ssf;
            scalar grad_z = meshSfPtr[3 * j + 2] * ssf;
#ifdef _OPENMP
#pragma omp atomic
#endif
            gradY_Species[owner * 3 + 0] += grad_x;
#ifdef _OPENMP
#pragma omp atomic
#endif
            gradY_Species[owner * 3 + 1] += grad_y;
#ifdef _OPENMP
#pragma omp atomic
#endif
            gradY_Species[owner * 3 + 2] += grad_z;
#ifdef _OPENMP
#pragma omp atomic
#endif
            gradY_Species[neighbor * 3 + 0] -= grad_x;
#ifdef _OPENMP
#pragma omp atomic
#endif
            gradY_Species[neighbor * 3 + 1] -= grad_y;
#ifdef _OPENMP
#pragma omp atomic
#endif
            gradY_Species[neighbor * 3 + 2] -= grad_z;
        }
#else

// OPT_FACE_SCHEDULE
        const MeshSchedule& schedule = MeshSchedule::getMeshSchedule();
        const labelList& face_scheduling = schedule.face_scheduling();

        #pragma omp parallel for
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
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
        }
        #pragma omp parallel for
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
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
        }
#endif


        gardY_face_time += clock.timeIncrement();

        for (label j = 0; j < nPatches; ++j) {
            label patchSize = patchSizes[j];
            const scalar* boundaryYi_patch = boundaryYi[i * nPatches + j];
            const scalar* boundaryYiInternal_patch = boundaryYiInternal[i * nPatches + j];
            const scalar* boundarySf_patch = boundarySf[j];
            const scalar* boundaryWeights_patch = boundaryWeights[j];

            const label* faceCells_patch = faceCells[j];

            if (patchTypes[j] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label k = 0; k < patchSize; ++k) {
                    scalar bouvf = boundaryYi_patch[k];
                    label cellIdx = faceCells_patch[k];

                    scalar bouSfx = boundarySf_patch[k * 3 + 0];
                    scalar bouSfy = boundarySf_patch[k * 3 + 1];
                    scalar bouSfz = boundarySf_patch[k * 3 + 2];

                    scalar grad_x = bouSfx * bouvf;
                    scalar grad_y = bouSfy * bouvf;
                    scalar grad_z = bouSfz * bouvf;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    gradY_Species[cellIdx * 3 + 0] += grad_x;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    gradY_Species[cellIdx * 3 + 1] += grad_y;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    gradY_Species[cellIdx * 3 + 2] += grad_z;
                }
            } else if (patchTypes[j] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label k = 0; k < patchSize; ++k) {
                    scalar bouWeight = boundaryWeights_patch[k];
                    label cellIdx = faceCells_patch[k];

                    scalar bouSfx = boundarySf_patch[k * 3 + 0];
                    scalar bouSfy = boundarySf_patch[k * 3 + 1];
                    scalar bouSfz = boundarySf_patch[k * 3 + 2];

                    scalar bouvf = (1 - bouWeight) * boundaryYi_patch[k] + bouWeight * boundaryYiInternal_patch[k];
                    scalar grad_x = bouSfx * bouvf;
                    scalar grad_y = bouSfy * bouvf;
                    scalar grad_z = bouSfz * bouvf;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    gradY_Species[cellIdx * 3 + 0] += grad_x;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    gradY_Species[cellIdx * 3 + 1] += grad_y;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    gradY_Species[cellIdx * 3 + 2] += grad_z;
                }
            }
        }

        gardY_boundary_time += clock.timeIncrement();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label c = 0; c < nCells; ++c) {
            scalar meshVRTmp = 1. / meshVPtr[c];;
            gradY_Species[c * 3 + 0] *= meshVRTmp;
            gradY_Species[c * 3 + 1] *= meshVRTmp;
            gradY_Species[c * 3 + 2] *= meshVRTmp;
        }

        gardY_div_time += clock.timeIncrement();

        for (label j = 0; j < nPatches; ++j) {

            label patchSize = patchSizes[j];

            scalar* boundaryGradY_patch = boundaryGradY[i * nPatches + j];
            scalar* boundaryGradYInternal_patch = boundaryGradYInternal[i * nPatches + j];
            const scalar* boundarySf_patch = boundarySf[j];
            const scalar* boundaryMagSf_patch = boundaryMagSf[j];
            
            const label* faceCells_patch = faceCells[j];

            if (patchTypes[j] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label k = 0; k < patchSize; ++k) {
                    label cellIdx = faceCells_patch[k];
                    scalar grad_x = gradY_Species[cellIdx * 3 + 0];
                    scalar grad_y = gradY_Species[cellIdx * 3 + 1];
                    scalar grad_z = gradY_Species[cellIdx * 3 + 2];
                    scalar n_x = boundarySf_patch[k * 3 + 0] / boundaryMagSf_patch[k];
                    scalar n_y = boundarySf_patch[k * 3 + 1] / boundaryMagSf_patch[k];
                    scalar n_z = boundarySf_patch[k * 3 + 2] / boundaryMagSf_patch[k];
                    scalar grad_correction = -(n_x * grad_x + n_y * grad_y + n_z * grad_z);
                    boundaryGradY_patch[k * 3 + 0] = grad_x + grad_correction * n_x;
                    boundaryGradY_patch[k * 3 + 1] = grad_y + grad_correction * n_y;
                    boundaryGradY_patch[k * 3 + 2] = grad_z + grad_correction * n_z;
                }
            } else if (patchTypes[j] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label k = 0; k < patchSize; ++k) {
                    label cellIdx = faceCells_patch[k];
                    boundaryGradYInternal_patch[k * 3 + 0] = gradY_Species[cellIdx * 3 + 0];
                    boundaryGradYInternal_patch[k * 3 + 1] = gradY_Species[cellIdx * 3 + 1];
                    boundaryGradYInternal_patch[k * 3 + 2] = gradY_Species[cellIdx * 3 + 2];
                }
            }
        }

        boundaryGardY_time += clock.timeIncrement();

        for (label j = 0; j < nPatches; ++j) {
            if (patchTypes[j] == MeshSchedule::PatchType::processor){
                MPI_Sendrecv(
                    boundaryGradYInternal[i * nPatches + j], 3 * patchSizes[j], MPI_DOUBLE, neighbProcNo[j], 0,
                    boundaryGradY[i * nPatches + j], 3 * patchSizes[j], MPI_DOUBLE, neighbProcNo[j], 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );
            }
        }

        boundaryGardY_comm_time += clock.timeIncrement();
    }

    Info << "preProcess_Y gardY_init_time : " << gardY_init_time << endl;
    Info << "preProcess_Y gardY_face_time : " << gardY_face_time << endl;
    Info << "preProcess_Y gardY_boundary_time : " << gardY_boundary_time << endl;
    Info << "preProcess_Y gardY_div_time : " << gardY_div_time << endl;
    Info << "preProcess_Y boundaryGardY_time : " << boundaryGardY_time << endl;
    Info << "preProcess_Y boundaryGardY_comm_time : " << boundaryGardY_comm_time << endl;
    
    // calculate sumYDiffError
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (label i = 0; i < nSpecies; ++i) {
        const scalar *gradY_Species = &gradY[i * nCells * 3];
        const scalar *rhoD_Species = rhoD[i];
#ifdef _OPENMP
#pragma omp for
#endif
        for (label j = 0; j < nCells; ++j) {
            sumYDiffErrorPtr[3 * j + 0] += gradY_Species[j * 3 + 0] * rhoD_Species[j];
            sumYDiffErrorPtr[3 * j + 1] += gradY_Species[j * 3 + 1] * rhoD_Species[j];
            sumYDiffErrorPtr[3 * j + 2] += gradY_Species[j * 3 + 2] * rhoD_Species[j];
        }

        for(label patchi = 0; patchi < nPatches; ++patchi){
            label patchSize = patchSizes[patchi];
            const scalar* boundaryGradY_patch = boundaryGradY[i * nPatches + patchi];
            const scalar* boundaryAlpha_patch = boundaryAlpha[patchi];
            scalar* boundarySumYDiffErrorPtr_patch = boundarySumYDiffErrorPtr[patchi];
            if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp for
#endif
                for (label k = 0; k < patchSize; ++k) {
                    boundarySumYDiffErrorPtr_patch[3 * k + 0] += boundaryGradY_patch[k * 3 + 0] * boundaryAlpha_patch[k];
                    boundarySumYDiffErrorPtr_patch[3 * k + 1] += boundaryGradY_patch[k * 3 + 1] * boundaryAlpha_patch[k];
                    boundarySumYDiffErrorPtr_patch[3 * k + 2] += boundaryGradY_patch[k * 3 + 2] * boundaryAlpha_patch[k];
                }
            } else if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp for
#endif
                for (label k = 0; k < patchSize; ++k) {
                    boundarySumYDiffErrorPtr_patch[3 * k + 0] += boundaryGradY_patch[k * 3 + 0] * boundaryAlpha_patch[k];
                    boundarySumYDiffErrorPtr_patch[3 * k + 1] += boundaryGradY_patch[k * 3 + 1] * boundaryAlpha_patch[k];
                    boundarySumYDiffErrorPtr_patch[3 * k + 2] += boundaryGradY_patch[k * 3 + 2] * boundaryAlpha_patch[k];
                }
            }
        }
    }

    Info << "preProcess_Y sumYDiffError : " << clock.timeIncrement() << endl;

    // calculate phiUc
    // internal
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

    Info << "preProcess_Y phiUc internal : " << clock.timeIncrement() << endl;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi){
        label patchSize = patchSizes[patchi];
        const scalar* boundarySf_patch = boundarySf[patchi];
        const scalar* boundarySumYDiffErrorPtr_patch = boundarySumYDiffErrorPtr[patchi];
        scalar* boundaryPhiUcPtr_patch = boundaryPhiUcPtr[patchi];
        if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label k = 0; k < patchSize; ++k) {
                scalar boundary_sfx = boundarySf_patch[k * 3 + 0];
                scalar boundary_sfy = boundarySf_patch[k * 3 + 1];
                scalar boundary_sfz = boundarySf_patch[k * 3 + 2];

                scalar boundary_ssfx = boundarySumYDiffErrorPtr_patch[3 * k + 0];
                scalar boundary_ssfy = boundarySumYDiffErrorPtr_patch[3 * k + 1];
                scalar boundary_ssfz = boundarySumYDiffErrorPtr_patch[3 * k + 2];

                boundaryPhiUcPtr_patch[k] = boundary_sfx * boundary_ssfx + boundary_sfy * boundary_ssfy + boundary_sfz * boundary_ssfz;
            }
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label k = 0; k < patchSize; ++k) {
                scalar boundary_sfx = boundarySf_patch[k * 3 + 0];
                scalar boundary_sfy = boundarySf_patch[k * 3 + 1];
                scalar boundary_sfz = boundarySf_patch[k * 3 + 2];
                scalar boundary_ssfx = boundarySumYDiffErrorPtr_patch[3 * k + 0];
                scalar boundary_ssfy = boundarySumYDiffErrorPtr_patch[3 * k + 1];
                scalar boundary_ssfz = boundarySumYDiffErrorPtr_patch[3 * k + 2];
                boundaryPhiUcPtr_patch[k] = boundary_sfx * boundary_ssfx + boundary_sfy * boundary_ssfy + boundary_sfz * boundary_ssfz;
            }
        }
    }

    Info << "preProcess_Y phiUc boundary : " << clock.timeIncrement() << endl;

    // calculate upwind weight
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label j = 0; j < nFaces; ++j) {
        upwindWeightsPtr[j] = phiPtr[j] >= 0 ? 1.: 0.;
    }

    Info << "preProcess_Y upwind : " << clock.timeIncrement() << endl;

    // free ptrs
    delete[] boundaryYi;
    for(label i = 0; i < nSpecies * nPatches; ++i){
        if(boundaryYiInternal[i] != nullptr){
            delete[] boundaryYiInternal[i];
        }
    }
    delete[] boundaryYiInternal;
    delete[] faceCells;
    delete[] boundarySf;
    delete[] boundaryMagSf;
    delete[] boundaryWeights;
    delete[] boundaryAlpha;
    for(label i = 0; i < nPatches; ++i){
        if(boundaryAlphaInternal[i] != nullptr){
            delete[] boundaryAlphaInternal[i];
        }
    }
    delete[] boundaryAlphaInternal;
    delete[] gradY;
    for(label i = 0; i < nPatches * nPatches; ++i){
        delete[] boundaryGradY[i];
        if(boundaryGradYInternal[i] != nullptr){
            delete[] boundaryGradYInternal[i];
        }
    }
    delete[] boundaryGradY;
    delete[] boundaryGradYInternal;
    delete[] YiPtr;
    delete[] boundarySumYDiffErrorPtr;
    delete[] rhoD;

    Info << "preProcess_Y free : " << clock.timeIncrement() << endl;
    Info << "preProcess_Y total : " << clock.elapsedTime() << endl;
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
    clockTime clock;
    assert(splitting == false);

    const fvMesh& mesh = Yi.mesh();
    assert(mesh.moving() == false);

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

    const MeshSchedule& meshSchedule = MeshSchedule::getMeshSchedule();
    const labelList& face_scheduling = meshSchedule.face_scheduling();
    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();

    // basic variable
    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const scalar *weightsPtr = &weights[0];
    const scalar *meshVPtr = &mesh.V()[0];
    const scalar *deltaCoeffsPtr = &mesh.deltaCoeffs()[0];
    const scalar *magSfPtr = &mesh.magSf()[0];

    Info << "Gen_Y init : " << clock.timeIncrement() << endl;

    // allocate memory
    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;

    constScalarPtr* boundary_phi = new constScalarPtr[nPatches];
    constScalarPtr* boundary_phiUc = new constScalarPtr[nPatches];
    constScalarPtr* boundaryWeights = new constScalarPtr[nPatches];
    constScalarPtr* boundary_delta_coeffs = new constScalarPtr[nPatches];
    constScalarPtr* boundary_mag_sf = new constScalarPtr[nPatches];
    constScalarPtr* boundary_rhoD = new constScalarPtr[nPatches];
    scalarPtr* boundary_rhoD_internal = new scalarPtr[nPatches];
    scalarPtr* boundary_upwind_weights = new scalarPtr[nPatches];
    constLabelPtr* faceCells = new constLabelPtr[nPatches];

    scalarPtr* internal_coeffs = new scalarPtr[nPatches];
    scalarPtr* boundary_coeffs = new scalarPtr[nPatches];
    scalarPtr* value_internal_coeffs = new scalarPtr[nPatches];
    scalarPtr* value_boundary_coeffs = new scalarPtr[nPatches];
    scalarPtr* gradient_internal_coeffs = new scalarPtr[nPatches];
    scalarPtr* gradient_boundary_coeffs = new scalarPtr[nPatches];

    const scalar *rhoPtr = &rho[0];
    const scalar *rhoOldTimePtr = &rho.oldTime()[0];
    const scalar *YiOldTimePtr = &Yi.oldTime()[0];
    const scalar *phiPtr = &phi[0];
    const scalar *phiUcPtr = &phiUc[0];
    const scalar *rhoDPtr = &rhoD[0];
    const scalar *suPtr = &su[0];

    for (label patchi = 0; patchi < nPatches; ++patchi) {
        label patchSize = patchSizes[patchi];
        faceCells[patchi] = mesh.boundary()[patchi].faceCells().begin(); // patchSize
        boundary_mag_sf[patchi] = mesh.magSf().boundaryField()[patchi].begin(); // patchSize
        boundaryWeights[patchi] = mesh.surfaceInterpolation::weights().boundaryField()[patchi].begin(); // patchSize
        boundary_delta_coeffs[patchi] = mesh.deltaCoeffs().boundaryField()[patchi].begin(); // patchSize

        const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
        boundary_phi[patchi] = patchPhi.begin(); // patchSize

        const fvsPatchScalarField& patchPhiUc = phiUc.boundaryField()[patchi];
        boundary_phiUc[patchi] = patchPhiUc.begin(); // patchSize

        const fvPatchScalarField& patchRhoD = rhoD.boundaryField()[patchi];
        boundary_rhoD[patchi] = patchRhoD.begin(); // patchSize

        internal_coeffs[patchi] = new scalar[patchSize];
        boundary_coeffs[patchi] = new scalar[patchSize];
        value_internal_coeffs[patchi] = new scalar[patchSize];
        value_boundary_coeffs[patchi] = new scalar[patchSize];
        gradient_internal_coeffs[patchi] = new scalar[patchSize];
        gradient_boundary_coeffs[patchi] = new scalar[patchSize];
        boundary_upwind_weights[patchi] = new scalar[patchSize];

        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            scalarField patchRhoDInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoD).patchInternalField()();
            boundary_rhoD_internal[patchi] = new scalar[patchSize];
            memcpy(boundary_rhoD_internal[patchi], patchRhoDInternal.begin(), patchSize * sizeof(scalar));
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
            boundary_rhoD_internal[patchi] = nullptr;
        } else {
            Info << "boundary condition not supported" << endl;
            std::exit(-1);
        }
    }

    Info << "Gen_Y allocate : " << clock.timeIncrement() << endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        for (label f = 0; f < patchSizes[patchi]; ++f) {
            boundary_upwind_weights[patchi][f] = boundary_phi[patchi][f] >= 0 ? 1.: 0.;
        }
    }

    Info << "Gen_Y upwind weight : " << clock.timeIncrement() << endl;

    // update boundary coeffs
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f){
                value_internal_coeffs[patchi][f] = 1.;
                value_boundary_coeffs[patchi][f] = 0.;
                gradient_internal_coeffs[patchi][f] = 0.;
                gradient_boundary_coeffs[patchi][f] = 0.;
            } 
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f){
                scalar bouDeltaCoeffs = boundary_delta_coeffs[patchi][f];
                scalar bouWeight = boundary_upwind_weights[patchi][f];
                value_internal_coeffs[patchi][f] = bouWeight;
                value_boundary_coeffs[patchi][f] = 1 - bouWeight;
                gradient_internal_coeffs[patchi][f] = -1 * bouDeltaCoeffs;
                gradient_boundary_coeffs[patchi][f] = bouDeltaCoeffs;
            }
        }    
    }

    Info << "Gen_Y update boundary coeffs : " << clock.timeIncrement() << endl;

    // ddt
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        diagPtr[i] += rDeltaT * rhoPtr[i] * meshVPtr[i];
        sourcePtr[i] += rDeltaT * rhoOldTimePtr[i] * YiOldTimePtr[i] * meshVPtr[i];
    }

    Info << "Gen_Y ddt : " << clock.timeIncrement() << endl;

    // fvmDiv(phi, Yi)
    // - internal
    for (label i = 0; i < nFaces; ++i) {
        scalar w = upwindWeights[i];
        scalar f = phiPtr[i];

        scalar lower_value = (-w) * f;
        scalar upper_value = (1 - w) * f;
        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;
    }

    // - boundary
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label f = 0; f < patchSizes[patchi]; ++f){
            scalar boundary_f = boundary_phi[patchi][f];
            internal_coeffs[patchi][f] = value_internal_coeffs[patchi][f] * boundary_f;
            boundary_coeffs[patchi][f] = -1. * value_boundary_coeffs[patchi][f] * boundary_f;
        }
    }

    Info << "Gen_Y div phi : " << clock.timeIncrement() << endl;

    // fvmDiv(phiUc, Yi)
    // - internal
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nFaces; ++i) {
        scalar w = upwindWeights[i];
        scalar f = phiUcPtr[i];

        scalar lower_value = (-w) * f;
        scalar upper_value = (1 - w) * f;
        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;
    }

    // - boundary
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label f = 0; f < patchSizes[patchi]; ++f){
            scalar boundary_f = boundary_phiUc[patchi][f];
            internal_coeffs[patchi][f] += boundary_f * value_internal_coeffs[patchi][f];
            boundary_coeffs[patchi][f] -= boundary_f * value_boundary_coeffs[patchi][f];
        }
    }

    Info << "Gen_Y div phiUc : " << clock.timeIncrement() << endl;

    // fvm::laplacian(DEff(), Yi)
    // - internal
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nFaces; ++i) {
        label owner = own[i];
        label neighbor = nei[i];
        scalar w = weightsPtr[i];
        scalar face_gamma = w * rhoDPtr[owner] + (1 - w) * rhoDPtr[neighbor];

        scalar upper_value = -1. * face_gamma * magSfPtr[i] * deltaCoeffsPtr[i];
        scalar lower_value = upper_value;

        upperPtr[i] += upper_value;
        lowerPtr[i] += lower_value;
    }

    // - boundary
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label f = 0; f < patchSizes[patchi]; ++f){
            scalar boundary_value = boundary_rhoD[patchi][f] * boundary_mag_sf[patchi][f];
            internal_coeffs[patchi][f] -= boundary_value * gradient_internal_coeffs[patchi][f];
            boundary_coeffs[patchi][f] += boundary_value * gradient_boundary_coeffs[patchi][f];
        }
    }
    Info << "Gen_Y laplacian DEff : " << clock.timeIncrement() << endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for (label j = face_start; j < face_end; ++j) {
            label owner = own[j];
            label neighbor = nei[j];
            diagPtr[owner] -= lowerPtr[j];
            diagPtr[neighbor] -= upperPtr[j];
        }
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
        label face_start = face_scheduling[face_scheduling_i]; 
        label face_end = face_scheduling[face_scheduling_i+1];
        for (label j = face_start; j < face_end; ++j) {
            label owner = own[j];
            label neighbor = nei[j];
            diagPtr[owner] -= lowerPtr[j];
            diagPtr[neighbor] -= upperPtr[j];
        }
    }

    Info << "Gen_Y diag : " << clock.timeIncrement() << endl;

    // add source
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        sourcePtr[i] += meshVPtr[i] * suPtr[i];
    }

    // cpoy result to fvm
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label patchi = 0; patchi < nPatches; ++patchi){
        for (label f = 0; f < patchSizes[patchi]; ++f){
            fvm.internalCoeffs()[patchi][f] = internal_coeffs[patchi][f];
            fvm.boundaryCoeffs()[patchi][f] = boundary_coeffs[patchi][f];
        }
    }

    Info << "Gen_Y copy back : " << clock.timeIncrement() << endl;
    

    // free ptrs
    delete[] boundary_phi;
    delete[] boundary_phiUc;
    delete[] boundaryWeights;
    delete[] boundary_delta_coeffs;
    delete[] boundary_mag_sf;
    delete[] boundary_rhoD;
    delete[] boundary_upwind_weights;
    delete[] faceCells;

    for(label i = 0; i < nPatches; ++i){
        if (boundary_rhoD_internal[i] != nullptr){
            delete[] boundary_rhoD_internal[i];
        }
        delete[] internal_coeffs[i];
        delete[] boundary_coeffs[i];
        delete[] value_internal_coeffs[i];
        delete[] value_boundary_coeffs[i];
        delete[] gradient_internal_coeffs[i];
        delete[] gradient_boundary_coeffs[i];
    }
    delete[] boundary_rhoD_internal;
    delete[] internal_coeffs;
    delete[] boundary_coeffs;
    delete[] value_internal_coeffs;
    delete[] value_boundary_coeffs;
    delete[] gradient_internal_coeffs;
    delete[] gradient_boundary_coeffs;

    Info << "Gen_Y free : " << clock.timeIncrement() << endl;
    Info << "Gen_Y total : " << clock.elapsedTime() << endl;

    return tfvm;
}

}