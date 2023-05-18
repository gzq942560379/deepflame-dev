/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2019 OpenFOAM Foundation
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

Class
    Foam::masterCoarsestHPCGProcAgglomeration

Description
    Processor agglomeration of HPCGAgglomerations.

SourceFiles
    masterCoarsestHPCGProcAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef masterCoarsestHPCGProcAgglomeration_H
#define masterCoarsestHPCGProcAgglomeration_H

#include "HPCGProcAgglomeration.H"
#include "DynamicList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class HPCGAgglomeration;

/*---------------------------------------------------------------------------*\
              Class masterCoarsestHPCGProcAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class masterCoarsestHPCGProcAgglomeration
:
    public HPCGProcAgglomeration
{
    // Private Data

        DynamicList<label> comms_;


public:

    //- Runtime type information
    TypeName("masterCoarsest");


    // Constructors

        //- Construct given agglomerator and controls
        masterCoarsestHPCGProcAgglomeration
        (
            HPCGAgglomeration& agglom,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        masterCoarsestHPCGProcAgglomeration
        (
            const masterCoarsestHPCGProcAgglomeration&
        );


    //- Destructor
    virtual ~masterCoarsestHPCGProcAgglomeration();


    // Member Functions

       //- Modify agglomeration. Return true if modified
        virtual bool agglomerate();

        //- Disallow default bitwise assignment


    // Member Operators

        void operator=(const masterCoarsestHPCGProcAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
