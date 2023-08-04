#pragma once

#include "tmp.H"
#include "dimensionedType.H"
#include "volFieldsFwd.H"
#include "surfaceFieldsFwd.H"
#include "typeInfo.H"
#include "runTimeSelectionTables.H"
#include "fvMatrices.H"
#include "fvMesh.H"
#include "turbulentFluidThermoModel.H"
#include "CombustionModel.H"
#include <mpi.h>
#include <algorithm>
#include "PstreamGlobals.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


// namespace fv
// {

// fvm::ddt
template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::ddt
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvm::div
template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
);

// fvc::div
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::grad
template<class Type>
tmp
<
    GeometricField
    <
        typename outerProduct<vector, Type>::type,
        fvPatchField,
        volMesh
    >
>
gaussGradSchemeGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf
);


template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacianUncorrected
(
    const surfaceScalarField& gammaMagSf,
    const surfaceScalarField& deltaCoeffs,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvm::laplacian
template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacian
(
    const GeometricField<scalar, fvPatchField, volMesh>& gammaScalarVol,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// turbulence->divDevRhoReff(U)
tmp<fvVectorMatrix>
turbulenceModelLinearViscousStressDivDevRhoReff
(
    volVectorField& U,
    compressible::turbulenceModel& turbulence
);

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence
);

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
    fv::convectionScheme<scalar>& mvConvection
);

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
);


void check_fvmatrix_equal(fvScalarMatrix& a,fvScalarMatrix& b);
void check_fvmatrix_equal(fvVectorMatrix& a,fvVectorMatrix& b);


} // End namespace Foam


// ************************************************************************* //
