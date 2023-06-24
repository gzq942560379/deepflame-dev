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

#include "geomRenumber.H"
#include "addToRunTimeSelectionTable.H"
#include "Random.H"
#include <vector>
#include <algorithm>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(geomRenumber, 0);

    addToRunTimeSelectionTable
    (
        renumberMethod,
        geomRenumber,
        dictionary
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::geomRenumber::geomRenumber(const dictionary& renumberDict)
:
    renumberMethod(renumberDict)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::labelList Foam::geomRenumber::renumber
(
    const pointField& points
) const
{
    

    std::vector<Point> Points(points.size());
    forAll(points, i){
        Points[i].id_ = i;
        Points[i].x_ = static_cast<label>(points[i][0] * 1e9);
        Points[i].y_ = static_cast<label>(points[i][1] * 1e9);
        Points[i].z_ = static_cast<label>(points[i][2] * 1e9);
    }    
    std::sort(Points.begin(), Points.end());
    // for(size_t i = 0; i < Points.size(); ++i){
    //     Info << i << " : " << Points[i].id_ << ", " << Points[i].x_ << ", " << Points[i].y_ << ", " << Points[i].z_ << endl;
    // }

    // labelList newToOld(points.size(), -1);

    // for(size_t i = 0; i < Points.size(); ++i){
    //     if(newToOld[Points[i].id_] != -1){
    //         FatalErrorInFunction
    //             << "Renumbering is not one-to-one. Both index "
    //             << newToOld[Points[i].id_]
    //             << " and " << i << " map onto " << Points[i].id_
    //             << " ." << endl
    //             << exit(FatalError);
    //     }
    //     newToOld[Points[i].id_] = i;
    // }

    labelList newToOld(points.size());
    for(size_t i = 0; i < Points.size(); ++i){
        newToOld[i] = Points[i].id_;
    }
    return newToOld;
}



Foam::labelList Foam::geomRenumber::renumber
(
    const polyMesh& mesh,
    const pointField& points
) const
{
    return renumber(points);
}


Foam::labelList Foam::geomRenumber::renumber
(
    const labelListList& cellCells,
    const pointField& points
) const
{
    return renumber(points);
}


// ************************************************************************* //
