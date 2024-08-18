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
#include <omp.h>
#include <mpi.h>
#include <yaml-cpp/yaml.h>
#include "PstreamGlobals.H"


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
    // rho_(mesh_.objectRegistry::lookupObject<volScalarField>("rho")),
    rho_(const_cast<volScalarField&>(dynamic_cast<rhoThermo&>(thermo).rho())),
    // mu_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).mu()())),
    // psi_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).psi())),
    mu_(const_cast<volScalarField&>(dynamic_cast<rhoThermo&>(thermo).mu()())),
    psi_(const_cast<volScalarField&>(dynamic_cast<rhoThermo&>(thermo).psi())),
    useThermoTranNN(false),
    mixfrac_
    (
        IOobject
        (
            "mixfrac",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimless, -1)
    ),
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
    // if (!Qdot_.typeHeaderOk<volScalarField>())
    // {
    //     useDNN = false;
    // }

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

    useThermoTranNN = this->lookupOrDefault("useThermoTranNN", false);
    if(useThermoTranNN)
        {
            call_ThermoTranNN = pybind11::module_::import("ThermoTranNN");
            Info << nl << "ThermoTranNN.py was loaded." << nl << endl;
        } 
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

    std::vector<string> speciesName = {"H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "CH3", 
            "CH4", "CO", "CO2", "HCO", "CH2O", "CH3O", "C2H6", "CH3O2", "N2"};
    for (auto name : speciesName)
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

    // forAll(hc_, i)
    // {
    //     hc_[i] = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
    // }

    // set normalization parameters

#ifdef USE_BLASDNN
    int mpisize, mpirank;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);

    if(flag_mpi_init){
        MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
        MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    }

    torchSwitch_ = this->subDict("TorchSettings").lookupOrDefault("torch", false);
    useDNN = true;
    useThermoTranNN = this->lookupOrDefault("useThermoTranNN", false);
    // if (!Qdot_.typeHeaderOk<volScalarField>())
    // {
    //     useDNN = false;
    // }
    if(torchSwitch_){
        BLASDNNModelPath_ = this->subDict("TorchSettings").lookupOrDefault("BLASDNNModelPath", string(""));
        DNNInferencer_blas_.load_models(BLASDNNModelPath_);

        int count;
        std::string norm_str;
        char* buffer;

        if (mpirank == 0 || !flag_mpi_init){
            std::ifstream fin(BLASDNNModelPath_ + "/norm.yaml");
            if (!fin) {
                SeriousError << "open norm error , norm path : " << BLASDNNModelPath_ + "/norm.yaml" << endl;
                MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
            }
            std::ostringstream oss;
            oss.str("");
            oss << fin.rdbuf();
            norm_str = oss.str();
            count = norm_str.size();
            fin.close();

            buffer = new char[count];
            std::copy(norm_str.begin(), norm_str.end(), buffer);
        }

        if(flag_mpi_init){
            MPI_Bcast(&count, 1, MPI_INT, 0, PstreamGlobals::MPI_COMM_FOAM);
            if (mpirank != 0)   
                buffer = new char[count];
            MPI_Bcast(buffer, count, MPI_CHAR, 0, PstreamGlobals::MPI_COMM_FOAM);
            if (mpirank != 0) {
                norm_str = std::string(buffer, count);
            }
            delete[] buffer;
        }

        // chemistry norm
        YAML::Node norm = YAML::Load(norm_str);
        YAML::Node XmuNode = norm["Xmu"];
        for (size_t i = 0; i < XmuNode.size(); i++){
            Xmu_.push_back(XmuNode[i].as<double>());
        }
        YAML::Node XstdNode = norm["Xstd"];
        for (size_t i = 0; i < XstdNode.size(); i++){
            Xstd_.push_back(XstdNode[i].as<double>());
        }
        YAML::Node YmuNode = norm["Ymu"];
        for (size_t i = 0; i < YmuNode.size(); i++){
            Ymu_.push_back(YmuNode[i].as<double>());
        }
        YAML::Node YstdNode = norm["Ystd"];
        for (size_t i = 0; i < YstdNode.size(); i++){
            Ystd_.push_back(YstdNode[i].as<double>());
        }

    }

    if(useThermoTranNN)
    {
        thermoDNNModelPath_ = this->subDict("TorchSettings").lookupOrDefault("thermoDNNModelPath", string(""));
        DNNThermo_blas_.load_models(thermoDNNModelPath_);
        Info << "ThermoDNN was loaded." << endl;

        int count;
        std::string thermo_norm_str;
        char* buffer;

        if (mpirank == 0 || !flag_mpi_init){
            // thermo norm
            std::ifstream fin = std::ifstream(thermoDNNModelPath_ + "/norm.yaml");
            if (!fin) {
                SeriousError << "open norm error , norm path : " << thermoDNNModelPath_ + "/norm.yaml" << endl;
                MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
            }
            std::ostringstream oss;
            oss.str("");         
            oss << fin.rdbuf();
            thermo_norm_str = oss.str();
            count = thermo_norm_str.size();
            fin.close();

            buffer = new char[count];
            std::copy(thermo_norm_str.begin(), thermo_norm_str.end(), buffer);
        }

        if(flag_mpi_init){
            MPI_Bcast(&count, 1, MPI_INT, 0, PstreamGlobals::MPI_COMM_FOAM);
            if (mpirank != 0)   
                buffer = new char[count];
            MPI_Bcast(buffer, count, MPI_CHAR, 0, PstreamGlobals::MPI_COMM_FOAM);
            if (mpirank != 0) {
                thermo_norm_str = std::string(buffer, count);
            }
            delete[] buffer;
        }

        // thermo norm
        YAML::Node thermoNorm = YAML::Load(thermo_norm_str);
        YAML::Node thermoMuNode = thermoNorm["mean"];

        for (size_t i = 0; i < thermoMuNode.size(); i++){
            thermomu_.push_back(thermoMuNode[i].as<double>());
        }
        YAML::Node thermoStdNode = thermoNorm["std"];
        for (size_t i = 0; i < thermoStdNode.size(); i++){
            thermostd_.push_back(thermoStdNode[i].as<double>());
        }
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
    // double correctThermo_part1_start, correctThermo_part1_end, correctThermo_part1_time = 0.;
    // double correctThermo_part2_start, correctThermo_part2_end, correctThermo_part2_time = 0.;
    // double correctThermo_start, correctThermo_end, correctThermo_time = 0.;
    // double correctThermo_part1_1_time = 0.;
    // double correctThermo_part1_2_time = 0.;
    // double correctThermo_part1_3_time = 0.;
    // double correctThermo_part1_4_time = 0.;
    // double correctThermo_part1_5_time = 0.;
    // double correctThermo_part1_6_time = 0.;
    // double correctThermo_part1_7_time = 0.;
    // double correctThermo_part1_8_time = 0.;
    
    // correctThermo_start = MPI_Wtime();

    // psi_.oldTime();
    // mixfrac_ = (16*(2*(Y_[8]/15+Y_[9]/16+Y_[10]/28+Y_[11]/44+Y_[12]/29+Y_[13]/30+Y_[14]/31+Y_[15]/15+Y_[16]/47)+0.5*(Y_[0]+Y_[1]+Y_[4]/17+Y_[5]/9+Y_[6]/33+Y_[7]/17+Y_[8]/5+Y_[9]/4+Y_[13]/15+3*Y_[14]/31+Y_[15]/5+3*Y_[16]/47)-(Y_[2]/16+Y_[3]/16+Y_[4]/17+Y_[5]/18+2*Y_[6]/33+Y_[7]/17+Y_[10]/28+Y_[11]/22+Y_[12]/29+Y_[13]/30+Y_[14]/31+2*Y_[16]/47))+1)/5;

    // correctThermo_part1_start = MPI_Wtime();
    // if(useThermoTranNN)
    // {   
    //     #ifdef USE_PYTORCH
    //     Info << "================using ThermoTranDNN=============" << endl;
    //     // psi_.oldTime();
    //     const scalarField& inputH = thermo_.he().primitiveField();
    //     const scalarField& inputP = p_.primitiveField();
    //     scalarField inputZ = mixfrac_.primitiveField();

    //     pybind11::array_t<double> vec6 = pybind11::array_t<double>({inputH.size()}, {8}, &inputH[0]); // cast vector to np.array
    //     pybind11::array_t<double> vec7 = pybind11::array_t<double>({inputP.size()}, {8}, &inputP[0]);
    //     pybind11::array_t<double> vec8 = pybind11::array_t<double>({inputZ.size()}, {8}, &inputZ[0]);
        
    //     Info <<  "vectors have all been constructed \n" << endl;
    //     pybind11::object result1 = call_ThermoTranNN.attr("useNet")(vec8, vec6, vec7);
    //     pybind11::array_t<double> result_array1(result1); 
    //     double* data_ptr1 = result_array1.mutable_data();
    //     forAll(T_, celli)  
    //     {
    //         rho_[celli] = data_ptr1[5*celli];
    //         T_[celli] = data_ptr1[5*celli+1];
    //         psi_[celli] = data_ptr1[5*celli+2];
    //         mu_[celli] = data_ptr1[5*celli+3];
    //         alpha_[celli] = data_ptr1[5*celli+4];
    //         forAll(rhoD_, i)
    //         {
    //             rhoD_[i][celli] = alpha_[celli];
    //         }
    //     }
    //     #endif 
    //     #ifdef USE_BLASDNN
    //     Info << "================using ThermoTranDNN=============" << endl;
    //     thermoDNN_blas(thermo_.he(), p_, mixfrac_, rho_, T_, psi_, mu_, alpha_, rhoD_);
    //     #endif
    // }
   
    // else
    // {
    //     forAll(T_, celli)
    //     {
    //         double tick0 = MPI_Wtime();
    //         scalarList yTemp(mixture_.nSpecies());
    //         scalarList dTemp(mixture_.nSpecies());
    //         scalarList hrtTemp(mixture_.nSpecies());
    //         forAll(Y_, i)
    //         {
    //             yTemp[i] = Y_[i][celli];
    //         }
    //         double tick1 = MPI_Wtime();
    //         correctThermo_part1_1_time += tick1 - tick0;

    //         CanteraGas_->setState_PY(p_[celli], yTemp.begin());
    //         double tick2 = MPI_Wtime();
    //         correctThermo_part1_2_time += tick2 - tick1;

    //         // cost
    //         CanteraGas_->setState_HP(thermo_.he()[celli], p_[celli]); // setState_HP needs (J/kg)
    //         double tick3 = MPI_Wtime();
    //         correctThermo_part1_3_time += tick3 - tick2;

    //         T_[celli] = CanteraGas_->temperature();
    //         double tick4 = MPI_Wtime();
    //         correctThermo_part1_4_time += tick4 - tick3;
    //         rho_[celli] = mixture_.rho(p_[celli],T_[celli]);
    //         // meanMolecularWeight() kg/kmol    RT() Joules/kmol
    //          psi_[celli] = mixture_.psi(p_[celli],T_[celli]);
    //         double tick5 = MPI_Wtime();
    //         correctThermo_part1_5_time += tick5 - tick4;

    //         // cost
    //         mu_[celli] = mixture_.CanteraTransport()->viscosity(); // Pa-s
    //         double tick6 = MPI_Wtime();
    //         correctThermo_part1_6_time += tick6 - tick5;

    //         alpha_[celli] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass()); // kg/(m*s)
    //         double tick7 = MPI_Wtime();
    //         correctThermo_part1_7_time += tick7 - tick6;
    //         // thermalConductivity() W/m/K
    //         // cp_mass()   J/kg/K

    //         // if (mixture_.transportModelName() == "UnityLewis")
    //         // {
    //             forAll(rhoD_, i)
    //             {
    //                 rhoD_[i][celli] = alpha_[celli];
    //             }
    //         // }
    //         double tick8 = MPI_Wtime();
    //         correctThermo_part1_8_time += tick8 - tick7;
    //     }
    // }
    // correctThermo_part1_end = MPI_Wtime();
    // correctThermo_part1_time += correctThermo_part1_end - correctThermo_part1_start;

    // const volScalarField::Boundary& pBf = p_.boundaryField();

    // volScalarField::Boundary& rhoBf = rho_.boundaryFieldRef();

    // volScalarField::Boundary& TBf = T_.boundaryFieldRef();

    // volScalarField::Boundary& psiBf = psi_.boundaryFieldRef();

    // volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();

    // volScalarField::Boundary& muBf = mu_.boundaryFieldRef();

    // volScalarField::Boundary& alphaBf = alpha_.boundaryFieldRef();

    // volScalarField::Boundary mixfracBf = mixfrac_.boundaryField();  

    // // volScalarField::Boundary& rhoDBf = rhoD_.boundaryFieldRef();
    // correctThermo_part2_start = MPI_Wtime();

    // forAll(T_.boundaryField(), patchi)
    // {
    //     const fvPatchScalarField& pp = pBf[patchi];
    //     fvPatchScalarField& prho = rhoBf[patchi];
    //     fvPatchScalarField& pT = TBf[patchi];
    //     fvPatchScalarField& ppsi = psiBf[patchi];
    //     fvPatchScalarField& ph = hBf[patchi];
    //     fvPatchScalarField& pmu = muBf[patchi];
    //     fvPatchScalarField& palpha = alphaBf[patchi];
    //     fvPatchScalarField pmixfrac = mixfracBf[patchi];

    //     if (pT.fixesValue())
    //     {
    //         Info << "==== pT.fixesValue() ===="<<endl;
    //         forAll(pT, facei)
    //         {
    //             forAll(Y_, i)
    //             {
    //                 yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
    //             }
    //             CanteraGas_->setState_TPY(pT[facei], pp[facei], yTemp_.begin());

    //             // prho[facei] = mixture_.rho(pp[facei],pT[facei]);

    //             ph[facei] = CanteraGas_->enthalpy_mass();

    //             ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();
    //             prho[facei] = mixture_.rho(pp[facei],pT[facei]);

    //             pmu[facei] = mixture_.CanteraTransport()->viscosity();

    //             palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

    //             forAll(rhoD_, i)
    //             {
    //                 rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
    //             }

    //         }
    //     }
    //     else
    //     {
    //         if(useThermoTranNN)
    //         {
    //             #ifdef USE_PYTORCH
    //             Info << "==== patch using Thermo DNN ===="<<endl;
    //             // scalarField inputZ = mixfrac_.primitiveField();
    //             pybind11::array_t<double> pvec8 = pybind11::array_t<double>({pmixfrac.size()}, {8}, &pmixfrac[0]);
    //             pybind11::array_t<double> pvec6 = pybind11::array_t<double>({ph.size()}, {8}, &ph[0]);
    //             pybind11::array_t<double> pvec7 = pybind11::array_t<double>({pp.size()}, {8}, &pp[0]);
    //             pybind11::object presult1 = call_ThermoTranNN.attr("useNet")(pvec8, pvec6, pvec7); // for density only
    //             pybind11::array_t<double> presult_array1(presult1);
    //             double* pdata_ptr1 = presult_array1.mutable_data();
    //             forAll(pT, facei) 
    //             {
    //                 prho[facei] = pdata_ptr1[5*facei];
    //                 pT[facei] = pdata_ptr1[5*facei+1];
    //                 ppsi[facei] = pdata_ptr1[5*facei+2];
    //                 pmu[facei] = pdata_ptr1[5*facei+3];
    //                 palpha[facei] = pdata_ptr1[5*facei+4];
    //                 forAll(rhoD_, i)
    //                 {
    //                     rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
    //                 }
    //             }
    //             #endif
    //         }
    //         else
    //         {
    //             forAll(pT, facei)
    //             {
    //                 forAll(Y_, i)
    //                 {
    //                     yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
    //                 }
    //                 CanteraGas_->setState_PY(pp[facei], yTemp_.begin());
    //                 CanteraGas_->setState_HP(ph[facei], pp[facei]);

    //                 pT[facei] = CanteraGas_->temperature();
    //                 prho[facei] = mixture_.rho(pp[facei],pT[facei]);

    //                 ppsi[facei] = mixture_.psi(pp[facei],pT[facei]);

    //                 pmu[facei] = mixture_.CanteraTransport()->viscosity();

    //                 palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

    //                 // if (mixture_.transportModelName() == "UnityLewis")
    //                 // {
    //                     forAll(rhoD_, i)
    //                     {
    //                         rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
    //                     }
    //                 // }
    //             }
    //         }
    //     }
    // }

    // correctThermo_part2_end = MPI_Wtime();
    // correctThermo_part2_time += correctThermo_part2_end - correctThermo_part2_start;
    
    // correctThermo_end = MPI_Wtime();
    // correctThermo_time += correctThermo_end - correctThermo_start;

    // Info << "correctThermo Total : " << correctThermo_time << endl;
    // Info << "correctThermo part1 : " << correctThermo_part1_time << endl;
    // Info << "correctThermo part2 : " << correctThermo_part2_time << endl;
    // Info << "correctThermo part1 1 : " << correctThermo_part1_1_time << endl;
    // Info << "correctThermo part1 2 : " << correctThermo_part1_2_time << endl;
    // Info << "correctThermo part1 3 : " << correctThermo_part1_3_time << endl;
    // Info << "correctThermo part1 4 : " << correctThermo_part1_4_time << endl;
    // Info << "correctThermo part1 5 : " << correctThermo_part1_5_time << endl;
    // Info << "correctThermo part1 6 : " << correctThermo_part1_6_time << endl;
    // Info << "correctThermo part1 7 : " << correctThermo_part1_7_time << endl;
    // Info << "correctThermo part1 8 : " << correctThermo_part1_8_time << endl;
    clockTime clock;

    psi_.oldTime();

    Info << "psi_.oldTime() : " <<  clock.timeIncrement() << endl;


    // label nCells = Y_[0].size();

    // assert(mixfrac_.size() == nCells);

    // typedef const scalar* constScalarPtr;
    // typedef scalar* scalarPtr;

    // scalarPtr mixfracPtr = mixfrac_.begin();
    // constScalarPtr Y0 = Y_[0].begin();
    // constScalarPtr Y1 = Y_[1].begin();
    // constScalarPtr Y2 = Y_[2].begin();
    // constScalarPtr Y3 = Y_[3].begin();
    // constScalarPtr Y4 = Y_[4].begin();
    // constScalarPtr Y5 = Y_[5].begin();
    // constScalarPtr Y6 = Y_[6].begin();
    // constScalarPtr Y7 = Y_[7].begin();
    // constScalarPtr Y8 = Y_[8].begin();
    // constScalarPtr Y9 = Y_[9].begin();
    // constScalarPtr Y10 = Y_[10].begin();
    // constScalarPtr Y11 = Y_[11].begin();
    // constScalarPtr Y12 = Y_[12].begin();
    // constScalarPtr Y13 = Y_[13].begin();
    // constScalarPtr Y14 = Y_[14].begin();
    // constScalarPtr Y15 = Y_[15].begin();
    // constScalarPtr Y16 = Y_[16].begin();

    // for(label c = 0; c < nCells; ++c){
    //     mixfracPtr[c] = (16.*(2.*(Y8[c]/15.+Y9[c]/16.+Y10[c]/28.+Y11[c]/44.+Y12[c]/29.+Y13[c]/30.+Y14[c]/31.+Y15[c]/15.+Y16[c]/47.)+0.5*(Y0[c]+Y1[c]+Y4[c]/17.+Y5[c]/9.+Y6[c]/33.+Y7[c]/17.+Y8[c]/5.+Y9[c]/4.+Y13[c]/15.+3.*Y14[c]/31.+Y15[c]/5.+3*Y16[c]/47.)-(Y2[c]/16.+Y3[c]/16.+Y4[c]/17.+Y5[c]/18.+2.*Y6[c]/33.+Y7[c]/17.+Y10[c]/28.+Y11[c]/22.+Y12[c]/29.+Y13[c]/30.+Y14[c]/31.+2.*Y16[c]/47.))+1.)/5.;
    // }
    
    mixfrac_ = (16*(2*(Y_[8]/15+Y_[9]/16+Y_[10]/28+Y_[11]/44+Y_[12]/29+Y_[13]/30+Y_[14]/31+Y_[15]/15+Y_[16]/47)+0.5*(Y_[0]+Y_[1]+Y_[4]/17+Y_[5]/9+Y_[6]/33+Y_[7]/17+Y_[8]/5+Y_[9]/4+Y_[13]/15+3*Y_[14]/31+Y_[15]/5+3*Y_[16]/47)-(Y_[2]/16+Y_[3]/16+Y_[4]/17+Y_[5]/18+2*Y_[6]/33+Y_[7]/17+Y_[10]/28+Y_[11]/22+Y_[12]/29+Y_[13]/30+Y_[14]/31+2*Y_[16]/47))+1)/5;
    Info << "mixfrac_ : " <<  clock.timeIncrement() << endl;

    if(useThermoTranNN)
    {   
        #ifdef USE_PYTORCH
        Info << "================using ThermoTranDNN=============" << endl;
        // psi_.oldTime();
        const scalarField& inputH = thermo_.he().primitiveField();
        const scalarField& inputP = p_.primitiveField();
        scalarField inputZ = mixfrac_.primitiveField();

        pybind11::array_t<double> vec6 = pybind11::array_t<double>({inputH.size()}, {8}, &inputH[0]); // cast vector to np.array
        pybind11::array_t<double> vec7 = pybind11::array_t<double>({inputP.size()}, {8}, &inputP[0]);
        pybind11::array_t<double> vec8 = pybind11::array_t<double>({inputZ.size()}, {8}, &inputZ[0]);
        
        Info <<  "vectors have all been constructed \n" << endl;
        pybind11::object result1 = call_ThermoTranNN.attr("useNet")(vec8, vec6, vec7);
        pybind11::array_t<double> result_array1(result1); 
        double* data_ptr1 = result_array1.mutable_data();
        forAll(T_, celli)  
        {
            rho_[celli] = data_ptr1[5*celli];
            T_[celli] = data_ptr1[5*celli+1];
            psi_[celli] = data_ptr1[5*celli+2];
            mu_[celli] = data_ptr1[5*celli+3];
            alpha_[celli] = data_ptr1[5*celli+4];
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = alpha_[celli];
            }
        }
        #endif 
        #ifdef USE_BLASDNN
        Info << "================using ThermoTranDNN=============" << endl;
        thermoDNN_blas(thermo_.he(), p_, mixfrac_, rho_, T_, psi_, mu_, alpha_, rhoD_);
        #endif
    }else{
        forAll(T_, celli)
        {
            scalarList yTemp(mixture_.nSpecies());
            scalarList dTemp(mixture_.nSpecies());
            scalarList hrtTemp(mixture_.nSpecies());
            forAll(Y_, i)
            {
                yTemp[i] = Y_[i][celli];
            }

            CanteraGas_->setState_PY(p_[celli], yTemp.begin());

            // cost
            CanteraGas_->setState_HP(thermo_.he()[celli], p_[celli]); // setState_HP needs (J/kg)

            T_[celli] = CanteraGas_->temperature();
            rho_[celli] = mixture_.rho(p_[celli],T_[celli]);
            // meanMolecularWeight() kg/kmol    RT() Joules/kmol
             psi_[celli] = mixture_.psi(p_[celli],T_[celli]);

            // cost
            mu_[celli] = mixture_.CanteraTransport()->viscosity(); // Pa-s

            alpha_[celli] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass()); // kg/(m*s)
            // thermalConductivity() W/m/K
            // cp_mass()   J/kg/K

            // if (mixture_.transportModelName() == "UnityLewis")
            // {
                forAll(rhoD_, i)
                {
                    rhoD_[i][celli] = alpha_[celli];
                }
            // }
        }
    }

    Info << "Thermo part 1 : " <<  clock.timeIncrement() << endl;

    const volScalarField::Boundary& pBf = p_.boundaryField();

    volScalarField::Boundary& rhoBf = rho_.boundaryFieldRef();

    volScalarField::Boundary& TBf = T_.boundaryFieldRef();

    volScalarField::Boundary& psiBf = psi_.boundaryFieldRef();

    volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();

    volScalarField::Boundary& muBf = mu_.boundaryFieldRef();

    volScalarField::Boundary& alphaBf = alpha_.boundaryFieldRef();

    volScalarField::Boundary mixfracBf = mixfrac_.boundaryField();  

    // volScalarField::Boundary& rhoDBf = rhoD_.boundaryFieldRef();

    forAll(T_.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        fvPatchScalarField& prho = rhoBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        fvPatchScalarField& ppsi = psiBf[patchi];
        fvPatchScalarField& ph = hBf[patchi];
        fvPatchScalarField& pmu = muBf[patchi];
        fvPatchScalarField& palpha = alphaBf[patchi];
        fvPatchScalarField pmixfrac = mixfracBf[patchi];

        if (pT.fixesValue())
        {
            Info << "==== pT.fixesValue() ===="<<endl;
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_TPY(pT[facei], pp[facei], yTemp_.begin());

                // prho[facei] = mixture_.rho(pp[facei],pT[facei]);

                ph[facei] = CanteraGas_->enthalpy_mass();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();
                prho[facei] = mixture_.rho(pp[facei],pT[facei]);

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                forAll(rhoD_, i)
                {
                    rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                }

            }
        }
        else
        {
            if(useThermoTranNN)
            {
                #ifdef USE_PYTORCH
                Info << "==== patch using Thermo DNN ===="<<endl;
                // scalarField inputZ = mixfrac_.primitiveField();
                pybind11::array_t<double> pvec8 = pybind11::array_t<double>({pmixfrac.size()}, {8}, &pmixfrac[0]);
                pybind11::array_t<double> pvec6 = pybind11::array_t<double>({ph.size()}, {8}, &ph[0]);
                pybind11::array_t<double> pvec7 = pybind11::array_t<double>({pp.size()}, {8}, &pp[0]);
                pybind11::object presult1 = call_ThermoTranNN.attr("useNet")(pvec8, pvec6, pvec7); // for density only
                pybind11::array_t<double> presult_array1(presult1);
                double* pdata_ptr1 = presult_array1.mutable_data();
                forAll(pT, facei) 
                {
                    prho[facei] = pdata_ptr1[5*facei];
                    pT[facei] = pdata_ptr1[5*facei+1];
                    ppsi[facei] = pdata_ptr1[5*facei+2];
                    pmu[facei] = pdata_ptr1[5*facei+3];
                    palpha[facei] = pdata_ptr1[5*facei+4];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                #endif
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
                    prho[facei] = mixture_.rho(pp[facei],pT[facei]);

                    ppsi[facei] = mixture_.psi(pp[facei],pT[facei]);

                    pmu[facei] = mixture_.CanteraTransport()->viscosity();

                    palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                    // if (mixture_.transportModelName() == "UnityLewis")
                    // {
                        forAll(rhoD_, i)
                        {
                            rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                        }
                    // }
                }
            }
        }
    }

    Info << "Thermo part 2 : " <<  clock.timeIncrement() << endl;
    Info << "Total correctThermo Time : " <<  clock.elapsedTime() << endl;
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

        timer.timeIncrement();
        auto guestProblems = balancer_.balance(allProblems);
        auto ownProblems = balancer_.getRemaining(allProblems);
        t_balance = timer.timeIncrement();

        timer.timeIncrement();
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        t_solveBuffer = timer.timeIncrement();

        timer.timeIncrement();
        incomingSolutions = balancer_.unbalance(guestSolutions);
        incomingSolutions.append(ownSolutions);
        t_unbalance = timer.timeIncrement();
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

#ifdef USE_BLASDNN
#include "blasdnnFunctions.H"
#include "thermodnnFunctions.H"
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
