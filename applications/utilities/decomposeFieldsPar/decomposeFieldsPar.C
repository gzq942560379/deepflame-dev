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

Application
    engineSwirl

Description
    Generates a swirling flow for engine calculations.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "mathematicalConstants.H"
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // read .txt
    std::vector<double *> filedata;
    std::ifstream infile;
    std::string str;
    infile.open("/home/runze/massiveParallel/deepflame-dev/examples/dfLowMachFoam/threeD_reactingTGV/H2/cvodeIntegrator_128/adiabatic_128.txt");
    while (std::getline(infile, str))
    {
        double * data = new double[128]{0.0};
        std::stringstream sumstr(str);
        string out;
        int i = 0;
        while (sumstr >> out)
        {
            double a = std::stod(out); 
            data[i] = a;
            i++;
        }
        filedata.push_back(data);
    }

    scalar x, y, z;
    label index;
    forAll(mesh.C(), celli)
    {
        // vector c = mesh.C()[celli];
        x = mesh.C()[celli][0];
        y = mesh.C()[celli][1];
        z = mesh.C()[celli][2];

        // set filed U
        U[celli] = vector(4*std::sin(x/0.001)*std::cos(y/0.001)*std::cos(z/0.001),-4*std::cos(x/0.001)*std::sin(y/0.001)*std::cos(z/0.001),0.0);

        // set scalar fields
        index = round((x/0.001) / 0.0494739);
        T[celli] = filedata[1][index];
        H2[celli] = filedata[2][index];
        H[celli] = filedata[3][index];
        O2[celli] = filedata[4][index];
        OH[celli] = filedata[5][index];
        O[celli] = filedata[6][index];
        H2O[celli] = filedata[7][index];
        HO2[celli] = filedata[8][index];
        H2O2[celli] = filedata[9][index];
        N2[celli] = filedata[10][index];
        
    }

    U.correctBoundaryConditions();
    T.correctBoundaryConditions();
    H2.correctBoundaryConditions();
    H.correctBoundaryConditions();
    O2.correctBoundaryConditions();
    OH.correctBoundaryConditions();
    O.correctBoundaryConditions();
    H2O.correctBoundaryConditions();
    HO2.correctBoundaryConditions();
    H2O2.correctBoundaryConditions();
    N2.correctBoundaryConditions();

    U.write();
    T.write();
    H2.write();
    H.write();
    O2.write();
    OH.write();
    O.write();
    H2O.write();
    HO2.write();
    H2O2.write();
    N2.write();

    Info<< "\n end\n";

    return 0;
}


// ************************************************************************* //
