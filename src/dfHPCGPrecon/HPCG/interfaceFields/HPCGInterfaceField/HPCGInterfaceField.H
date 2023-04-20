/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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
    Foam::HPCGInterfaceField

Description
     Abstract base class for HPCG agglomerated interface fields.

SourceFiles
    HPCGInterfaceField.C
    HPCGInterfaceFieldNew.C

\*---------------------------------------------------------------------------*/

#ifndef HPCGInterfaceField_H
#define HPCGInterfaceField_H

#include "lduInterfaceField.H"
#include "HPCGInterface.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class HPCGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class HPCGInterfaceField
:
    public lduInterfaceField
{
    // Private Data

        //- Local reference cast into the interface
        const HPCGInterface& interface_;


public:

    //- Runtime type information
    TypeName("HPCGInterfaceField");


    // Declare run-time constructor selection tables

        declareRunTimeSelectionTable
        (
            autoPtr,
            HPCGInterfaceField,
            lduInterfaceField,
            (
                const HPCGInterface& HPCGCp,
                const lduInterfaceField& fineInterface
            ),
            (HPCGCp, fineInterface)
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            HPCGInterfaceField,
            lduInterface,
            (
                const HPCGInterface& HPCGCp,
                const bool doTransform,
                const int rank
            ),
            (HPCGCp, doTransform, rank)
        );


    // Selectors

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<HPCGInterfaceField> New
        (
            const HPCGInterface& HPCGCp,
            const lduInterfaceField& fineInterface
        );

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<HPCGInterfaceField> New
        (
            const HPCGInterface& HPCGCp,
            const bool doTransform,
            const int rank
        );


    // Constructors

        //- Construct from HPCG interface and fine level interface field
        HPCGInterfaceField
        (
            const HPCGInterface& HPCGCp,
            const lduInterfaceField&
        )
        :
            lduInterfaceField(HPCGCp),
            interface_(HPCGCp)
        {}

        //- Construct from HPCG interface and fine level interface field
        HPCGInterfaceField
        (
            const HPCGInterface& HPCGCp,
            const bool doTransform,
            const int rank
        )
        :
            lduInterfaceField(HPCGCp),
            interface_(HPCGCp)
        {}

        //- Disallow default bitwise copy construction
        HPCGInterfaceField(const HPCGInterfaceField&) = delete;


    // Member Functions

        // Access

            //- Return interface
            const HPCGInterface& interface() const
            {
                return interface_;
            }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const HPCGInterfaceField&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
