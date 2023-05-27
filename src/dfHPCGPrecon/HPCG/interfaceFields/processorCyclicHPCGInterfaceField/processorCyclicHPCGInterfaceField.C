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

#include "processorCyclicHPCGInterfaceField.H"
#include "addToRunTimeSelectionTable.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(processorCyclicHPCGInterfaceField, 0);
    addToRunTimeSelectionTable
    (
        HPCGInterfaceField,
        processorCyclicHPCGInterfaceField,
        lduInterface
    );
    addToRunTimeSelectionTable
    (
        HPCGInterfaceField,
        processorCyclicHPCGInterfaceField,
        lduInterfaceField
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::processorCyclicHPCGInterfaceField::processorCyclicHPCGInterfaceField
(
    const HPCGInterface& HPCGCp,
    const lduInterfaceField& fineInterface
)
:
    processorHPCGInterfaceField(HPCGCp, fineInterface)
{}


Foam::processorCyclicHPCGInterfaceField::processorCyclicHPCGInterfaceField
(
    const HPCGInterface& HPCGCp,
    const bool doTransform,
    const int rank
)
:
    processorHPCGInterfaceField(HPCGCp, doTransform, rank)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::processorCyclicHPCGInterfaceField::~processorCyclicHPCGInterfaceField()
{}


// ************************************************************************* //
