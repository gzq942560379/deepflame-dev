/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
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

#include "CanteraMixture.H"
#include "fvMesh.H"
#include <iostream>
#include <fstream>
#include <string>
#include "PstreamGlobals.H"

#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/kinetics/Kinetics.h"
#include "cantera/kinetics/KineticsFactory.h"
#include "cantera/transport/TransportBase.h"
#include "cantera/transport/TransportFactory.h"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::CanteraMixture::CanteraMixture
(
    const dictionary& thermoDict,
    const fvMesh& mesh,
    const word& phaseName
)
:
    CanteraTorchProperties_
    (
        IOobject
        (
            "CanteraTorchProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    CanteraMechanismFile_(CanteraTorchProperties_.lookup("CanteraMechanismFile")),
    CanteraSolution_(Cantera::newSolution(CanteraMechanismFile_, "")),
    CanteraGas_(CanteraSolution_->thermo()),
    transportModelName_(CanteraTorchProperties_.lookup("transportModel")),
    CanteraTransport_(newTransportMgr(transportModelName_, CanteraGas_.get())),
    Y_(nSpecies()),
    Tref_(mesh.objectRegistry::lookupObject<volScalarField>("T")),
    pref_(mesh.objectRegistry::lookupObject<volScalarField>("p")),
    yTemp_(nSpecies()),
    HaTemp_(nSpecies()),
    CpTemp_(nSpecies()),
    CvTemp_(nSpecies()),
    muTemp_(nSpecies())
{
    // buildCanteraSolution();

    Y_.resize(nSpecies());
    yTemp_.resize(nSpecies());
    HaTemp_.resize(nSpecies());
    CpTemp_.resize(nSpecies());
    CvTemp_.resize(nSpecies());
    muTemp_.resize(nSpecies());

    forAll(Y_, i)
    {
        species_.append(CanteraGas_->speciesName(i));
    }

    tmp<volScalarField> tYdefault;

    forAll(Y_, i)
    {
        IOobject header
        (
            species_[i],
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ
        );

        // check if field exists and can be read
        if (header.typeHeaderOk<volScalarField>(true))
        {
            Y_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                        species_[i],
                        mesh.time().timeName(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh
                )
            );
        }
        else
        {
            // Read Ydefault if not already read
            if (!tYdefault.valid())
            {
                word YdefaultName("Ydefault");

                IOobject timeIO
                (
                    YdefaultName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                IOobject constantIO
                (
                    YdefaultName,
                    mesh.time().constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                IOobject time0IO
                (
                    YdefaultName,
                    Time::timeName(0),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                if (timeIO.typeHeaderOk<volScalarField>(true))
                {
                    tYdefault = new volScalarField(timeIO, mesh);
                }
                else if (constantIO.typeHeaderOk<volScalarField>(true))
                {
                    tYdefault = new volScalarField(constantIO, mesh);
                }
                else
                {
                    tYdefault = new volScalarField(time0IO, mesh);
                }
            }

            Y_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                        species_[i],
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    tYdefault()
                )
            );
        }
    }


}

void Foam::CanteraMixture::buildCanteraSolution(){
    // assert(CanteraMechanismFile_ end with .yaml);

    int32_t count;
    char* buffer;    
    std::string CanteraMechanism;

    int mpisize, mpirank;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);

    if(flag_mpi_init){
        MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
        MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    }

    if(mpirank == 0){
        std::ifstream fin(CanteraMechanismFile_);
        if (!fin) {
            SeriousError << "open CanteraMechanismFile_ error , CanteraMechanismFile_ : " << CanteraMechanismFile_ << endl;
            MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
        }
        std::ostringstream oss;
        oss << fin.rdbuf();
        CanteraMechanism = oss.str();
        count = CanteraMechanism.size();
        buffer = new char[count];
        std::copy(CanteraMechanism.begin(), CanteraMechanism.end(), buffer);
        fin.close();
    }

    MPI_Bcast(&count, 1, MPI_INT, 0, PstreamGlobals::MPI_COMM_FOAM);

    if(mpirank != 0){
        buffer = new char[count];
    }

    MPI_Bcast(buffer, count, MPI_CHAR, 0, PstreamGlobals::MPI_COMM_FOAM);

    if(mpirank != 0){
        CanteraMechanism = string(buffer, count);
    }

    delete[] buffer;

    // std::cout << CanteraMechanism << std::endl;

    Cantera::AnyMap root = Cantera::AnyMap::fromYamlString(CanteraMechanism);
    Cantera::AnyMap& phase = root["phases"].getMapWhere("name", "");

    Cantera::newSolution(CanteraMechanismFile_, "");
    // instantiate Solution object
    CanteraSolution_ = Cantera::Solution::create();
    // thermo phase
    CanteraSolution_->setThermo(std::shared_ptr<Cantera::ThermoPhase>(Cantera::newPhase(phase, root)));
    // kinetics
    std::vector<Cantera::ThermoPhase*> phases;
    phases.push_back(CanteraSolution_->thermo().get());
    CanteraSolution_->setKinetics(Cantera::newKinetics(phases, phase, root));
    // transport
    CanteraSolution_->setTransport(std::shared_ptr<Cantera::Transport>(Cantera::newDefaultTransportMgr(CanteraSolution_->thermo().get())));

    CanteraGas_ = CanteraSolution_->thermo();

    CanteraTransport_ = std::shared_ptr<Cantera::Transport>(newTransportMgr(transportModelName_, CanteraGas_.get()));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::CanteraMixture::read(const dictionary& thermoDict)
{
    //mixture_ = ThermoType(thermoDict.subDict("mixture"));

}


const Foam::CanteraMixture& Foam::CanteraMixture::cellMixture(const label celli) const
{
    forAll(Y_, i)
    {
        yTemp_[i] = Y_[i][celli];
    }
    CanteraGas_->setState_TPY(Tref_[celli], pref_[celli], yTemp_.begin());

    return *this;
}


const Foam::CanteraMixture& Foam::CanteraMixture::patchFaceMixture
(
    const label patchi,
    const label facei
) const
{
    forAll(Y_, i)
    {
        yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
    }
    CanteraGas_->setState_TPY(Tref_.boundaryField()[patchi][facei],pref_.boundaryField()[patchi][facei],yTemp_.begin());

    return *this;
}


Foam::scalar Foam::CanteraMixture::THE
(
    const scalar& h,
    const scalar& p,
    const scalar& T
) const
{
    CanteraGas_->setState_HP(h, p);
    return CanteraGas_->temperature();
}

Foam::scalar Foam::CanteraMixture::Hc() const
{
    scalar chemicalEnthalpy = 0;
    forAll(yTemp_, i)
    {
        chemicalEnthalpy += yTemp_[i]*CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
    }
    return chemicalEnthalpy;
}

// ************************************************************************* //
