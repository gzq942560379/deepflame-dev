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
    Foam::manualHPCGProcAgglomeration

Description
    Manual processor agglomeration of HPCGAgglomerations.

    In the HPCG control dictionary:

        processorAgglomerator manual;
        // List of level+procagglomeration where
        // procagglomeration is a set of labelLists. Each labelList is
        // a cluster of processor which gets combined onto the first element
        // in the list.
        processorAgglomeration
        (
            (
                3           // at level 3
                (
                    (0 1)   // coarse 0 from 0,1 (and moved onto 0)
                    (3 2)   // coarse 1 from 2,3 (and moved onto 3)
                )
            )
            (
                6           // at level6
                (
                    (0 1)   // coarse 0 from 0,1 (and moved onto 0)
                )
            )
        );

SourceFiles
    manualHPCGProcAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef manualHPCGProcAgglomeration_H
#define manualHPCGProcAgglomeration_H

#include "HPCGProcAgglomeration.H"
#include "DynamicList.H"
#include "Tuple2.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class HPCGAgglomeration;

/*---------------------------------------------------------------------------*\
              Class manualHPCGProcAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class manualHPCGProcAgglomeration
:
    public HPCGProcAgglomeration
{
    // Private Data

        //- Per level the agglomeration map
        const List<Tuple2<label, List<labelList>>> procAgglomMaps_;

        //- Any allocated communicators
        DynamicList<label> comms_;


public:

    //- Runtime type information
    TypeName("manual");


    // Constructors

        //- Construct given agglomerator and controls
        manualHPCGProcAgglomeration
        (
            HPCGAgglomeration& agglom,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        manualHPCGProcAgglomeration
        (
            const manualHPCGProcAgglomeration&
        );


    //- Destructor
    virtual ~manualHPCGProcAgglomeration();


    // Member Functions

       //- Modify agglomeration. Return true if modified
        virtual bool agglomerate();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const manualHPCGProcAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
