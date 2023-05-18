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
    Foam::noneHPCGProcAgglomeration

Description
    Processor agglomeration of HPCGAgglomerations.

SourceFiles
    noneHPCGProcAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef noneHPCGProcAgglomeration_H
#define noneHPCGProcAgglomeration_H

#include "HPCGProcAgglomeration.H"
#include "DynamicList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class HPCGAgglomeration;

/*---------------------------------------------------------------------------*\
              Class noneHPCGProcAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class noneHPCGProcAgglomeration
:
    public HPCGProcAgglomeration
{

public:

    //- Runtime type information
    TypeName("noneHPCGProcAgglomeration");


    // Constructors

        //- Construct given agglomerator and controls
        noneHPCGProcAgglomeration
        (
            HPCGAgglomeration& agglom,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        noneHPCGProcAgglomeration
        (
            const noneHPCGProcAgglomeration&
        );


    //- Destructor
    virtual ~noneHPCGProcAgglomeration();


    // Member Functions

       //- Modify agglomeration. Return true if modified
        virtual bool agglomerate()
        {
            return false;
        }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const noneHPCGProcAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
