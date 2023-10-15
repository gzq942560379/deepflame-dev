/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "UniformField.H"
#include "clockTime.H"
#include "runtime_assert.H"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::dfChemistryModel<ThermoType>::dfChemistryModel
(
    ThermoType& thermo
)
:
    IOdictionary
    (
        IOobject
        (
            "CanteraTorchProperties",
            thermo.db().time().constant(),
            thermo.db(),
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    thermo_(thermo),
    mixture_(dynamic_cast<CanteraMixture&>(thermo)),
    CanteraGas_(mixture_.CanteraGas()),
    mesh_(thermo.p().mesh()),
    chemistry_(lookup("chemistry")),
    relTol_(this->subDict("odeCoeffs").lookupOrDefault("relTol",1e-9)),
    absTol_(this->subDict("odeCoeffs").lookupOrDefault("absTol",1e-15)),
    Y_(mixture_.Y()),
    rhoD_(mixture_.nSpecies()),
    hai_(mixture_.nSpecies()),
    hc_(mixture_.nSpecies()),
    yTemp_(mixture_.nSpecies()),
    dTemp_(mixture_.nSpecies()),
    hrtTemp_(mixture_.nSpecies()),
    cTemp_(mixture_.nSpecies()),
    RR_(mixture_.nSpecies()),
    alpha_(const_cast<volScalarField&>(thermo.alpha())),
    T_(thermo.T()),
    p_(thermo.p()),
    rho_(mesh_.objectRegistry::lookupObject<volScalarField>("rho")),
    mu_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).mu()())),
    psi_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).psi())),
    Qdot_
    (
        IOobject
        (
            "Qdot",
            mesh_.time().timeName(),
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
    ),
    selectDNN_
    (
        IOobject
        (
            "selectDNN",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimless, -1)
    ),
    balancer_(createBalancer()),
    cpuTimes_
    (
        IOobject
        (
            "cellCpuTimes",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        scalar(0.0)
    )
{

#if defined USE_LIBTORCH || defined USE_PYTORCH
    useDNN = true;
    if (!Qdot_.typeHeaderOk<volScalarField>())
    {
        useDNN = false;
    }

    torchSwitch_ = this->subDict("TorchSettings").lookupOrDefault("torch", false);
    gpu_ = this->subDict("TorchSettings").lookupOrDefault("GPU", false),
    gpulog_ = this->subDict("TorchSettings").lookupOrDefault("log", false),

    time_allsolve_ = 0;
    time_submaster_ = 0;
    time_sendProblem_ = 0;
    time_RecvProblem_ = 0;
    time_sendRecvSolution_ = 0;
    time_getDNNinputs_ = 0;
    time_DNNinference_ = 0;
    time_updateSolutionBuffer_ = 0;
    time_getProblems_ = 0;
    time_cvode_ = 0;
    time_update_RR_ = 0;
#endif

#ifdef USE_LIBTORCH
    torchModelName1_ = this->subDict("TorchSettings").lookupOrDefault("torchModel1", word(""));
    torchModelName2_ = this->subDict("TorchSettings").lookupOrDefault("torchModel2", word(""));
    torchModelName3_ = this->subDict("TorchSettings").lookupOrDefault("torchModel3", word(""));

    // set the number of cores slaved by each GPU card
    cores_ = this->subDict("TorchSettings").lookupOrDefault("coresPerGPU", 8);
    GPUsPerNode_ = this->subDict("TorchSettings").lookupOrDefault("GPUsPerNode", 4);

    // initialization the Inferencer (if use multi GPU)
    if(torchSwitch_)
    {
        if (gpu_)
        {
            if(!(Pstream::myProcNo() % cores_)) // Now is a master
            {
                torch::jit::script::Module torchModel1_ = torch::jit::load(torchModelName1_);
                torch::jit::script::Module torchModel2_ = torch::jit::load(torchModelName2_);
                torch::jit::script::Module torchModel3_ = torch::jit::load(torchModelName3_);
                std::string device_;
                int CUDANo = (Pstream::myProcNo() / cores_) % GPUsPerNode_;
                device_ = "cuda:" + std::to_string(CUDANo);
                DNNInferencer DNNInferencer(torchModel1_, torchModel2_, torchModel3_, device_);
                DNNInferencer_ = DNNInferencer;
            }
        }
        else
        {
            torch::jit::script::Module torchModel1_ = torch::jit::load(torchModelName1_);
            torch::jit::script::Module torchModel2_ = torch::jit::load(torchModelName2_);
            torch::jit::script::Module torchModel3_ = torch::jit::load(torchModelName3_);
            std::string device_;
            device_ = "cpu";
            DNNInferencer DNNInferencer(torchModel1_, torchModel2_, torchModel3_, device_);
            DNNInferencer_ = DNNInferencer;
        }
    }
#endif

#ifdef USE_PYTORCH
    cores_ = this->subDict("TorchSettings").lookupOrDefault("coresPerNode", 8);

    time_vec2ndarray_ = 0;
    time_python_ = 0;
    time_python_import_module_ = 0;
    time_python_inference_ = 0;
#endif

#if defined USE_LIBTORCH || defined USE_PYTORCH
    // if use torch, create new communicator for solving cvode
    if (torchSwitch_)
    {
        labelList subRank;
        for (int rank = 0; rank < Pstream::nProcs(); rank ++)
        {
            if (rank % cores_)
            {
                subRank.append(rank);
            }
        }
        cvodeComm = UPstream::allocateCommunicator(UPstream::worldComm, subRank, true);
        if(Pstream::myProcNo() % cores_)
        {
            label sub_rank;
            MPI_Comm_rank(PstreamGlobals::MPICommunicators_[cvodeComm], &sub_rank);
            // std::cout<<"my ProcessNo in worldComm = " << Pstream::myProcNo() << ' '
            // << "my ProcessNo in cvodeComm = "<<Pstream::myProcNo(cvodeComm)<<std::endl;
        }
    }
#endif

    for(const auto& name : CanteraGas_->speciesNames())
    {
        species_.append(name);
    }
    forAll(RR_, fieldi)
    {
        RR_.set
        (
            fieldi,
            new volScalarField::Internal
            (
                IOobject
                (
                    "RR." + Y_[fieldi].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimMass/dimVolume/dimTime, 0)
            )
        );
    }

    forAll(rhoD_, i)
    {
        rhoD_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "rhoD_" + Y_[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimDensity*dimViscosity, 0) // kg/m/s
            )
        );
    }

    forAll(hai_, i)
    {
        hai_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "hai_" + Y_[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimEnergy/dimMass, 0)
            )
        );
    }
    if(balancer_.log())
    {
        cpuSolveFile_ = logFile("cpu_solve.out");
        cpuSolveFile_() << "                  time" << tab
                        << "           getProblems" << tab
                        << "           updateState" << tab
                        << "               balance" << tab
                        << "           solveBuffer" << tab
                        << "             unbalance" << tab
                        << "               rank ID" << endl;
    }

    Info<<"--- I am here in Cantera-construct ---"<<endl;
    Info<<"relTol_ === "<<relTol_<<endl;
    Info<<"absTol_ === "<<absTol_<<endl;

    forAll(hc_, i)
    {
        hc_[i] = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
    }
    // set normalization parameters
#if defined USE_TENSORFLOW || defined USE_BLASDNN
    Xmu0_ = {956.4666683951323,
                              1.2621251609602075,
                              -8.482865855078037,
                              -8.60195200775564,
                              -7.5687249938092975,
                              -8.739604352829021,
                              -3.0365348658864555,
                              -4.044646973729736,
                              -0.12868046894653598};
    Xstd0_ = {144.56082979138094,
                               0.4316114858005481,
                               1.3421800304159297,
                               1.3271564927376922,
                               1.964747648182199,
                               1.1993472911833807,
                               1.2594695379275647,
                               1.3518816605077604,
                               0.17392016053354714};
    Ymu0_ = {8901.112679962635,
                              27135.624769093312,
                              30141.97503208172,
                              24712.755148584696,
                              -372.9651472886253,
                              -493.34322699725413,
                              -4.31138850114707e-12};
    Ystd0_ = {8901.112679962635,
                               27135.624769093312,
                               30141.97503208172,
                               24712.755148584696,
                               372.96514728862553,
                               493.3432269972544,
                               9.409165181242247e-11};
    Xmu1_ = {1933.118541482812,
                              1.2327983023706526,
                              -5.705591538151852,
                              -6.446971251373195,
                              -4.169802387800032,
                              -6.1200334699867165,
                              -4.266343396329115,
                              -2.6007437468608616,
                              -0.4049762774428252};
    Xstd1_ = {716.6568054751183,
                               0.43268544913281914,
                               2.0857655247141387,
                               2.168997234412133,
                               2.707064105162402,
                               2.2681157746245897,
                               2.221785173612795,
                               1.5510851480805254,
                               0.30283229364455927};
    Ymu1_ = {175072.98234441387,
                              125434.41067566245,
                              285397.9376620931,
                              172924.8443087139,
                              -97451.53428068386,
                              -7160.953630852251,
                              -9.791262408691773e-10};
    Ystd1_ = {179830.51132577812,
                               256152.83860126554,
                               285811.9455262339,
                               263600.5448448552,
                               98110.53711881173,
                               11752.979335965118,
                               4.0735353885293555e-09};
    Xmu2_ = {2717.141719004927,
                              1.2871371577864235,
                              -5.240181052513087,
                              -4.8947914078286345,
                              -3.117070179161789,
                              -4.346362771443917,
                              -4.657258124450032,
                              -4.537442872141596,
                              -0.11656950757756744};
    Xstd2_ = {141.48030419772115,
                               0.4281422992061657,
                               0.6561518672685264,
                               0.9820405777881894,
                               1.0442969662425572,
                               0.7554583907448359,
                               1.7144519099198097,
                               1.1299391466695952,
                               0.15743252221610685};
    Ymu2_ = {-611.0636921032669,
                              -915.1244682112174,
                              519.5930550881994,
                              -11.949500174512165,
                              -2660.9187297995336,
                              159.56360614662788,
                              -7.136459430073843e-11};
    Ystd2_ = {611.0636921032669,
                               915.1244682112174,
                               519.5930550881994,
                               342.3100987934528,
                               2754.8463649064784,
                               313.3717647966624,
                               2.463374792192512e-10};


    torchSwitch_ = this->subDict("TorchSettings").lookupOrDefault("torch", false);
    useDNN = true;
    if (!Qdot_.typeHeaderOk<volScalarField>())
    {
        useDNN = false;
    }
#endif


#ifdef USE_TENSORFLOW
    if(torchSwitch_){
        tfModelPath0_ = this->subDict("TorchSettings").lookupOrDefault("tfModelPath0", string(""));
        tfModelPath1_ = this->subDict("TorchSettings").lookupOrDefault("tfModelPath1", string(""));
        tfModelPath2_ = this->subDict("TorchSettings").lookupOrDefault("tfModelPath2", string(""));

        std::ifstream input_file0(tfModelPath0_, std::ios::binary);
        std::cout<<"tfModelPath0_ = "<<tfModelPath0_<<std::endl;
        std::vector<char> model_data0((std::istreambuf_iterator<char>(input_file0)), (std::istreambuf_iterator<char>()));
        std::cout<<"model_data0 = "<<model_data0.size()<<std::endl;
        input_file0.close();

        std::ifstream input_file1(tfModelPath1_, std::ios::binary);
        std::vector<char> model_data1((std::istreambuf_iterator<char>(input_file1)), (std::istreambuf_iterator<char>()));
        std::cout<<"model_data1 = "<<model_data0.size()<<std::endl;
        input_file1.close();

        std::ifstream input_file2(tfModelPath2_, std::ios::binary);
        std::vector<char> model_data2((std::istreambuf_iterator<char>(input_file2)), (std::istreambuf_iterator<char>()));
        std::cout<<"model_data2 = "<<model_data0.size()<<std::endl;
        input_file2.close();

        DNNInferencertf DNNInferencertf(model_data0, model_data1, model_data2);
        DNNInferencertf_ = DNNInferencertf;
    }
#endif

#ifdef USE_BLASDNN
    if(torchSwitch_){
        BLASDNNModelPath_ = this->subDict("TorchSettings").lookupOrDefault("BLASDNNModelPath", string(""));
        DNNInferencer_blas_.load_models(BLASDNNModelPath_);
        // DNNInferencer_blas_.alloc_buffer(mesh_.nCells());
    }
#endif
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::dfChemistryModel<ThermoType>::
~dfChemistryModel()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


template<class ThermoType>
template<class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const DeltaTType& deltaT
)
{
    scalar result = 0;

#if defined(USE_LIBTORCH) || defined(USE_PYTORCH)
    if(torchSwitch_)
    {
        if (useDNN)
        {
            result = solve_DNN(deltaT);
        }
        else
        {
            result = solve_CVODE(deltaT);
            useDNN = true;
        }
    }
    else
    {
        result = solve_CVODE(deltaT);
    }
#elif defined(USE_TENSORFLOW)
    if(torchSwitch_)
    {
        if (useDNN)
        {
            result = solve_DNN_tf(deltaT);
        }
        else
        {
            result = solve_CVODE(deltaT);
            useDNN = true;
        }
    }
    else
    {
        result = solve_CVODE(deltaT);
    }
#elif defined(USE_BLASDNN)
    if(torchSwitch_)
    {
        if (useDNN)
        {
            result = solve_DNN_blas(deltaT);
        }
        else
        {
            result = solve_CVODE(deltaT);
            useDNN = true;
        }
    }
    else
    {
        result = solve_CVODE(deltaT);
    }
#else
    result = solve_CVODE(deltaT);
#endif

    return result;
}

template<class ThermoType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const scalar deltaT
)
{
    // Don't allow the time-step to change more than a factor of 2
    return min
    (
        this->solve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        2*deltaT
    );
}


template<class ThermoType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const scalarField& deltaT
)
{
    return this->solve<scalarField>(deltaT);
}


template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::setNumerics(Cantera::ReactorNet &sim)
{
    sim.setTolerances(relTol_,absTol_);
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::correctEnthalpy()
{
    forAll(T_, celli)
    {
        forAll(Y_, i)
        {
            yTemp_[i] = Y_[i][celli];
        }
        CanteraGas_->setState_TPY(T_[celli], p_[celli], yTemp_.begin());
        thermo_.he()[celli] = CanteraGas_->enthalpy_mass();
    }
    volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();
    volScalarField::Boundary& TBf = T_.boundaryFieldRef();
    const volScalarField::Boundary& pBf = p_.boundaryField();

    forAll(T_.boundaryField(), patchi)
    {
        fvPatchScalarField& ph = hBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        const fvPatchScalarField& pp = pBf[patchi];

        forAll(pT, facei)
        {
            forAll(Y_, i)
            {
                yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
            }
            CanteraGas_->setState_TPY(pT[facei], pp[facei], yTemp_.begin());
            ph[facei] = CanteraGas_->enthalpy_mass();
        }
    }
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::correctThermo()
{
    double correctThermo_part1_start, correctThermo_part1_end, correctThermo_part1_time = 0.;
    double correctThermo_part2_start, correctThermo_part2_end, correctThermo_part2_time = 0.;
    double correctThermo_start, correctThermo_end, correctThermo_time = 0.;
    double correctThermo_part1_1_time = 0.;
    double correctThermo_part1_2_time = 0.;
    double correctThermo_part1_3_time = 0.;
    double correctThermo_part1_4_time = 0.;
    double correctThermo_part1_5_time = 0.;
    double correctThermo_part1_6_time = 0.;
    double correctThermo_part1_7_time = 0.;
    double correctThermo_part1_8_time = 0.;

    correctThermo_start = MPI_Wtime();

    psi_.oldTime();

    correctThermo_part1_start = MPI_Wtime();

    forAll(T_, celli)
    {
        double tick0 = MPI_Wtime();
        scalarList yTemp(mixture_.nSpecies());
        scalarList dTemp(mixture_.nSpecies());
        scalarList hrtTemp(mixture_.nSpecies());
        forAll(Y_, i)
        {
            yTemp[i] = Y_[i][celli];
        }
        double tick1 = MPI_Wtime();
        correctThermo_part1_1_time += tick1 - tick0;

        CanteraGas_->setState_PY(p_[celli], yTemp.begin());
        double tick2 = MPI_Wtime();
        correctThermo_part1_2_time += tick2 - tick1;

        // cost
        CanteraGas_->setState_HP(thermo_.he()[celli], p_[celli]); // setState_HP needs (J/kg)
        double tick3 = MPI_Wtime();
        correctThermo_part1_3_time += tick3 - tick2;

        T_[celli] = CanteraGas_->temperature();
        double tick4 = MPI_Wtime();
        correctThermo_part1_4_time += tick4 - tick3;

        // meanMolecularWeight() kg/kmol    RT() Joules/kmol
        psi_[celli] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();
        double tick5 = MPI_Wtime();
        correctThermo_part1_5_time += tick5 - tick4;

        // cost
        mu_[celli] = mixture_.CanteraTransport()->viscosity(); // Pa-s
        double tick6 = MPI_Wtime();
        correctThermo_part1_6_time += tick6 - tick5;

        alpha_[celli] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass()); // kg/(m*s)
        double tick7 = MPI_Wtime();
        correctThermo_part1_7_time += tick7 - tick6;
        // thermalConductivity() W/m/K
        // cp_mass()   J/kg/K

        if (mixture_.transportModelName() == "UnityLewis")
        {
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = alpha_[celli];
            }
        }
        else
        {
            mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp.begin()); // m2/s

            CanteraGas_->getEnthalpy_RT(hrtTemp.begin()); //hrtTemp_=m_h0_RT non-dimension
            // constant::physicoChemical::R.value()   J/(mol·k)
            const scalar RT = constant::physicoChemical::R.value()*1e3*T_[celli]; // J/kmol/K
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = rho_[celli]*dTemp[i];

                // CanteraGas_->molecularWeight(i)    kg/kmol
                hai_[i][celli] = hrtTemp[i]*RT/CanteraGas_->molecularWeight(i);
            }
        }
        double tick8 = MPI_Wtime();
        correctThermo_part1_8_time += tick8 - tick7;
    }

    correctThermo_part1_end = MPI_Wtime();
    correctThermo_part1_time += correctThermo_part1_end - correctThermo_part1_start;

    const volScalarField::Boundary& pBf = p_.boundaryField();

    const volScalarField::Boundary& rhoBf = rho_.boundaryField();

    volScalarField::Boundary& TBf = T_.boundaryFieldRef();

    volScalarField::Boundary& psiBf = psi_.boundaryFieldRef();

    volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();

    volScalarField::Boundary& muBf = mu_.boundaryFieldRef();

    volScalarField::Boundary& alphaBf = alpha_.boundaryFieldRef();

    correctThermo_part2_start = MPI_Wtime();

    forAll(T_.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        const fvPatchScalarField& prho = rhoBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        fvPatchScalarField& ppsi = psiBf[patchi];
        fvPatchScalarField& ph = hBf[patchi];
        fvPatchScalarField& pmu = muBf[patchi];
        fvPatchScalarField& palpha = alphaBf[patchi];

        if (pT.fixesValue())
        {
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_TPY(pT[facei], pp[facei], yTemp_.begin());

                ph[facei] = CanteraGas_->enthalpy_mass();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                if (mixture_.transportModelName() == "UnityLewis")
                {
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                else
                {
                    mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin());

                    CanteraGas_->getEnthalpy_RT(hrtTemp_.begin());
                    const scalar RT = constant::physicoChemical::R.value()*1e3*pT[facei];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = prho[facei]*dTemp_[i];

                        hai_[i].boundaryFieldRef()[patchi][facei] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
                    }
                }
            }
        }
        else
        {
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_PY(pp[facei], yTemp_.begin());
                CanteraGas_->setState_HP(ph[facei], pp[facei]);

                pT[facei] = CanteraGas_->temperature();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                if (mixture_.transportModelName() == "UnityLewis")
                {
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                else
                {
                    mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin());

                    CanteraGas_->getEnthalpy_RT(hrtTemp_.begin());
                    const scalar RT = constant::physicoChemical::R.value()*1e3*pT[facei];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = prho[facei]*dTemp_[i];

                        hai_[i].boundaryFieldRef()[patchi][facei] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
                    }
                }
            }
        }
    }

    correctThermo_part2_end = MPI_Wtime();
    correctThermo_part2_time += correctThermo_part2_end - correctThermo_part2_start;
    
    correctThermo_end = MPI_Wtime();
    correctThermo_time += correctThermo_end - correctThermo_start;

    Info << "correctThermo Total : " << correctThermo_time << endl;
    Info << "correctThermo part1 : " << correctThermo_part1_time << endl;
    Info << "correctThermo part2 : " << correctThermo_part2_time << endl;
    Info << "correctThermo part1 1 : " << correctThermo_part1_1_time << endl;
    Info << "correctThermo part1 2 : " << correctThermo_part1_2_time << endl;
    Info << "correctThermo part1 3 : " << correctThermo_part1_3_time << endl;
    Info << "correctThermo part1 4 : " << correctThermo_part1_4_time << endl;
    Info << "correctThermo part1 5 : " << correctThermo_part1_5_time << endl;
    Info << "correctThermo part1 6 : " << correctThermo_part1_6_time << endl;
    Info << "correctThermo part1 7 : " << correctThermo_part1_7_time << endl;
    Info << "correctThermo part1 8 : " << correctThermo_part1_8_time << endl;
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::solveSingle
(
    ChemistryProblem& problem, ChemistrySolution& solution
)
{

    // Timer begins
    clockTime time;
    time.timeIncrement();

    Cantera::Reactor react;
    const scalar Ti = problem.Ti;
    const scalar pi = problem.pi;
    const scalar rhoi = problem.rhoi;
    const scalarList yPre_ = problem.Y;
    scalar Qdoti_ = 0;

    CanteraGas_->setState_TPY(Ti, pi, yPre_.begin());

    react.insert(mixture_.CanteraSolution());
    // keep T const before and after sim.advance. this will give you a little improvement
    react.setEnergy(0);
    Cantera::ReactorNet sim;
    sim.addReactor(react);
    setNumerics(sim);

    sim.advance(problem.deltaT);

    CanteraGas_->getMassFractions(yTemp_.begin());

    for (int i=0; i<mixture_.nSpecies(); i++)
    {
        solution.RRi[i] = (yTemp_[i] - yPre_[i]) / problem.deltaT * rhoi;
        Qdoti_ -= hc_[i]*solution.RRi[i];
    }

    // Timer ends
    solution.cpuTime = time.timeIncrement();
    solution.Qdoti = Qdoti_;

    solution.cellid = problem.cellid;
    solution.local = problem.local;
}



template <class ThermoType>
template<class DeltaTType>
Foam::DynamicList<Foam::ChemistryProblem>
Foam::dfChemistryModel<ThermoType>::getProblems
(
    const DeltaTType& deltaT
)
{
    const scalarField& T = T_;
    const scalarField& p = p_;

    DynamicList<ChemistryProblem> solved_problems(p.size(), ChemistryProblem(mixture_.nSpecies()));

    forAll(T, celli)
    {
        {
            for(label i = 0; i < mixture_.nSpecies(); i++)
            {
                yTemp_[i] = Y_[i][celli];
            }

            CanteraGas_->setState_TPY(T[celli], p[celli], yTemp_.begin());
            CanteraGas_->getConcentrations(cTemp_.begin());

            ChemistryProblem problem;
            problem.Y = yTemp_;
            problem.Ti = T[celli];
            problem.pi = p[celli];
            problem.rhoi = rho_[celli];
            problem.deltaT = deltaT[celli];
            problem.cpuTime = cpuTimes_[celli];
            problem.cellid = celli;

            solved_problems[celli] = problem;
        }

    }

    return solved_problems;
}


template <class ThermoType>
Foam::DynamicList<Foam::ChemistrySolution>
Foam::dfChemistryModel<ThermoType>::solveList
(
    UList<ChemistryProblem>& problems
)
{
    DynamicList<ChemistrySolution> solutions(
        problems.size(), ChemistrySolution(mixture_.nSpecies()));

    for(label i = 0; i < problems.size(); ++i)
    {
        solveSingle(problems[i], solutions[i]);
    }
    return solutions;
}


template <class ThermoType>
Foam::RecvBuffer<Foam::ChemistrySolution>
Foam::dfChemistryModel<ThermoType>::solveBuffer
(
    RecvBuffer<ChemistryProblem>& problems
)
{
    // allocate the solutions buffer
    RecvBuffer<ChemistrySolution> solutions;

    for(auto& p : problems)
    {
        solutions.append(solveList(p));
    }
    return solutions;
}



template <class ThermoType>
Foam::scalar
Foam::dfChemistryModel<ThermoType>::updateReactionRates
(
    const RecvBuffer<ChemistrySolution>& solutions,
    DynamicList<ChemistrySolution>& submasterODESolutions
)
{
    scalar deltaTMin = great;

    for(const auto& array : solutions)
    {
        for(const auto& solution : array)
        {
            if (solution.local)
            {
                for(label j = 0; j < mixture_.nSpecies(); j++)
                {
                    RR_[j][solution.cellid] = solution.RRi[j];
                }
                Qdot_[solution.cellid] = solution.Qdoti;

                cpuTimes_[solution.cellid] = solution.cpuTime;
            }
            else
            {
                submasterODESolutions.append(solution);
            }
        }
    }

    return deltaTMin;
}



template <class ThermoType>
Foam::LoadBalancer
Foam::dfChemistryModel<ThermoType>::createBalancer()
{
    const IOdictionary chemistryDict_tmp
        (
            IOobject
            (
                "CanteraTorchProperties",
                thermo_.db().time().constant(),
                thermo_.db(),
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );

    return LoadBalancer(chemistryDict_tmp);
}


template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_CVODE
(
    const DeltaTType& deltaT
)
{
    Info<<"=== begin solve_CVODE === "<<endl;
    // CPU time analysis
    clockTime timer;
    scalar t_getProblems(0);
    scalar t_updateState(0);
    scalar t_balance(0);
    scalar t_solveBuffer(0);
    scalar t_unbalance(0);

    if(!chemistry_)
    {
        return great;
    }

    timer.timeIncrement();
    DynamicList<ChemistryProblem> allProblems = getProblems(deltaT);
    t_getProblems = timer.timeIncrement();

    RecvBuffer<ChemistrySolution> incomingSolutions;

    if(balancer_.active())
    {
        Info<<"Now DLB algorithm is used!!"<<endl;
        timer.timeIncrement();
        balancer_.updateState(allProblems);
        t_updateState = timer.timeIncrement();

        Info<<"solve_CVODE 1"<<endl;

        timer.timeIncrement();
        auto guestProblems = balancer_.balance(allProblems);
        Info<<"solve_CVODE 2"<<endl;
        auto ownProblems = balancer_.getRemaining(allProblems);
        t_balance = timer.timeIncrement();
        Info<<"solve_CVODE 3"<<endl;

        timer.timeIncrement();
        auto ownSolutions = solveList(ownProblems);
        Info<<"solve_CVODE 4"<<endl;
        auto guestSolutions = solveBuffer(guestProblems);
        t_solveBuffer = timer.timeIncrement();
        Info<<"solve_CVODE 5"<<endl;

        timer.timeIncrement();
        incomingSolutions = balancer_.unbalance(guestSolutions);
        incomingSolutions.append(ownSolutions);
        t_unbalance = timer.timeIncrement();
        Info<<"solve_CVODE 6"<<endl;
    }
    else
    {
        Info<<"Now DLB algorithm is not used!!"<<endl;
        timer.timeIncrement();
        incomingSolutions.append(solveList(allProblems));
        t_solveBuffer = timer.timeIncrement();
    }

    if(balancer_.log())
    {
        balancer_.printState();
        cpuSolveFile_() << setw(22)
                        << this->time().timeOutputValue()<<tab
                        << setw(22) << t_getProblems<<tab
                        << setw(22) << t_updateState<<tab
                        << setw(22) << t_balance<<tab
                        << setw(22) << t_solveBuffer<<tab
                        << setw(22) << t_unbalance<<tab
                        << setw(22) << Pstream::myProcNo()
                        << endl;
    }
    DynamicList<ChemistrySolution> List;
    Info<<"=== end solve_CVODE === "<<endl;
    return updateReactionRates(incomingSolutions, List);
}

#ifdef USE_TENSORFLOW
#include "tensorflowFunctions.H"
#endif

#ifdef USE_BLASDNN
#include "blasdnnFunctions.H"
#endif

#if defined USE_LIBTORCH || defined USE_PYTORCH
#include "torchFunctions.H"
#endif


#ifdef USE_LIBTORCH
#include "libtorchFunctions.H"
#endif


#ifdef USE_PYTORCH
#include "pytorchFunctions.H"
#endif

// ************************************************************************* //
