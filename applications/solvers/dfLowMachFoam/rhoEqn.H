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

Global
    rhoEqn

Description
    Solve the continuity for density.

\*---------------------------------------------------------------------------*/
double rho_total_start, rho_total_end, rho_total_time = 0.;
double rho_build_start, rho_build_end, rho_build_time = 0.;
double rho_solve_start, rho_solve_end, rho_solve_time = 0.;

rho_total_start = MPI_Wtime();
{
    rho_build_start = MPI_Wtime();
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
    rho_build_end = MPI_Wtime();
    rho_build_time = rho_build_end - rho_build_start;

    rho_solve_start = MPI_Wtime();
    rhoEqn.solve();
    rho_solve_end = MPI_Wtime();
    rho_solve_time += rho_solve_end - rho_solve_start;
}
rho_total_end = MPI_Wtime();
rho_total_time += rho_total_end - rho_total_start;

Info << "rho total Time : " << rho_total_time << endl;
Info << "rho build Time : " << rho_build_time << endl;
Info << "rho solve Time : " << rho_solve_time << endl;
// ************************************************************************* //
