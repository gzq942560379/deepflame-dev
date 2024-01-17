#include "createRDeltaT.H"

double createFildes_tick_0 = MPI_Wtime();

IOdictionary CanteraTorchProperties
(
    IOobject
    (
        "CanteraTorchProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

double createFildes_tick_1 = MPI_Wtime();
Info << "createFildes_time_1 : " << createFildes_tick_1 - createFildes_tick_0 << endl;

Info<< "Reading thermophysical properties\n" << endl;

fluidThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;

double createFildes_tick_2 = MPI_Wtime();
Info << "createFildes_time_2 : " << createFildes_tick_2 - createFildes_tick_1 << endl;

thermo.validate(args.executable(), "ha");

double createFildes_tick_3 = MPI_Wtime();
Info << "createFildes_time_3 : " << createFildes_tick_3 - createFildes_tick_2 << endl;

volScalarField& T = thermo.T();

const volScalarField& psi = thermo.psi();
volScalarField& p = thermo.p();

double createFildes_tick_4 = MPI_Wtime();
Info << "createFildes_time_4 : " << createFildes_tick_4 - createFildes_tick_3 << endl;

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    thermo.rho()
);

double createFildes_tick_5 = MPI_Wtime();
Info << "createFildes_time_5 : " << createFildes_tick_5 - createFildes_tick_4 << endl;

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

double createFildes_tick_6 = MPI_Wtime();
Info << "createFildes_time_6 : " << createFildes_tick_6 - createFildes_tick_5 << endl;

#include "compressibleCreatePhi.H"

double createFildes_tick_7 = MPI_Wtime();
Info << "createFildes_time_7 : " << createFildes_tick_7 - createFildes_tick_6 << endl;

pressureControl pressureControl(p, rho, pimple.dict(), false);

double createFildes_tick_8 = MPI_Wtime();
Info << "createFildes_time_8 : " << createFildes_tick_8 - createFildes_tick_7 << endl;

mesh.setFluxRequired(p.name());

double createFildes_tick_9 = MPI_Wtime();
Info << "createFildes_time_9 : " << createFildes_tick_9 - createFildes_tick_8 << endl;

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

double createFildes_tick_10 = MPI_Wtime();
Info << "createFildes_time_10 : " << createFildes_tick_10 - createFildes_tick_9 << endl;

Info<< "Creating field dpdt\n" << endl;
volScalarField dpdt
(
    IOobject
    (
        "dpdt",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dpdt",p.dimensions()/dimTime, 0)
);
double createFildes_tick_11 = MPI_Wtime();
Info << "createFildes_time_11 : " << createFildes_tick_11 - createFildes_tick_10 << endl;

Info<< "Creating reaction model\n" << endl;
autoPtr<CombustionModel<basicThermo>> combustion
(
    CombustionModel<basicThermo>::New(thermo, turbulence())
);
Info<< "end Creating reaction model\n" << endl;
double createFildes_tick_12 = MPI_Wtime();
Info << "createFildes_time_12 : " << createFildes_tick_12 - createFildes_tick_11 << endl;

//const word combModelName(combustion->type());
const word combModelName(mesh.objectRegistry::lookupObject<IOdictionary>("combustionProperties").lookup("combustionModel"));
Info << "Combustion Model Name is confirmed as "<< combModelName << endl;

double createFildes_tick_13 = MPI_Wtime();
Info << "createFildes_time_13 : " << createFildes_tick_13 - createFildes_tick_12 << endl;

dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();

double createFildes_tick_14 = MPI_Wtime();
Info << "createFildes_time_14" << createFildes_tick_14 - createFildes_tick_13 << endl;

const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);

double createFildes_tick_15 = MPI_Wtime();
Info << "createFildes_time_15 : " << createFildes_tick_15 - createFildes_tick_14 << endl;

// Info<< "At initial time, min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

//for dpdt

Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

double createFildes_tick_16 = MPI_Wtime();
Info << "createFildes_time_16 : " << createFildes_tick_16 - createFildes_tick_15 << endl;

multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

if(combModelName!="flareFGM")
{
forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());
}

double createFildes_tick_17 = MPI_Wtime();
Info << "createFildes_time_17 : " << createFildes_tick_17 - createFildes_tick_16 << endl;

const scalar Sct = chemistry->lookupOrDefault("Sct", 1.);
volScalarField diffAlphaD
(
    IOobject
    (
        "diffAlphaD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimEnergy/dimTime/dimVolume, 0)
);

double createFildes_tick_18 = MPI_Wtime();
Info << "createFildes_time_18 : " << createFildes_tick_18 - createFildes_tick_17 << endl;

volVectorField hDiffCorrFlux
(
    IOobject
    (
        "hDiffCorrFlux",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector(dimensionSet(1,0,-3,0,0,0,0), Zero)
);

surfaceScalarField phiUc
(
    IOobject
    (
        "phiUc",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(1,0,-1,0,0,0,0), 0)
);

surfaceScalarField upwindWeights
(
    IOobject
    (
        "upwindWeights",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(0,0,0,0,0,0,0), 0)
);

double createFildes_tick_19 = MPI_Wtime();
Info << "createFildes_time_19 : " << createFildes_tick_19 - createFildes_tick_18 << endl;

volVectorField sumYDiffError
(
    IOobject
    (
        "sumYDiffError",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("sumYDiffError", dimDynamicViscosity/dimLength, Zero)
);

double createFildes_tick_20 = MPI_Wtime();
Info << "createFildes_time_20 : " << createFildes_tick_20 - createFildes_tick_19 << endl;

const Switch splitting = CanteraTorchProperties.lookupOrDefault("splittingStrategy", false);
// switch for field initialization
bool initialized = false;
bool renumberMesh = false;
int refineLevel = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("refineLevel", 0);

double createFildes_tick_21 = MPI_Wtime();
Info << "createFildes_time_21 : " << createFildes_tick_21 - createFildes_tick_20 << endl;

const surfaceScalarField linear_weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)

#ifdef USE_PYTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
#ifdef USE_LIBTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
