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

#include "dfMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>
#include "env.H"
#include "dfLduMatrix.H"
#include "dfCSRMatrix.H"
#include "dfBlockMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(dfMatrix, 1);
}

const Foam::label Foam::dfMatrix::solver::defaultMaxIter_ = 1000;

Foam::InnerMatrixFormat Foam::dfMatrix::getInnerMatrixTypeFromEnv(){
    const char* tmp = std::getenv("DFMATRIX_INNERMATRIX_TYPE");
    if(tmp == NULL){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU;
    }
    if(std::strcmp(tmp, "LDU") == 0){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU;
    }else if(std::strcmp(tmp, "CSR") == 0){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR;
    }else if(std::strcmp(tmp, "BLOCK_CSR") == 0){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR;
    }else{
        SeriousError << "Invalid DFMATRIX_INNERMATRIX_TYPE: " << tmp << endl << flush;
        std::exit(1);
    }
}

Foam::dfMatrix::dfMatrix(const lduMatrix& ldu):lduMatrix_(ldu)
{
    // innerMatrixPtr_ = new dfLduMatrix(ldu);
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfLduMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfCSRMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfCSRMatrix>(ldu);
            break;
        default:
            // error:
            SeriousError << "Invalid InnerMatrixFormat: " << format << endl << flush;
            std::exit(1);
    }
}

Foam::dfMatrix::dfMatrix(const lduMatrix& ldu, const labelList& regionPtr):lduMatrix_(ldu)
{
    // innerMatrixPtr_ = new dfLduMatrix(ldu);
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfLduMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfCSRMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR:
            Info << "Building Block CSR matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfBlockMatrix>(ldu, regionPtr);
            break;
        default:
            // error:
            SeriousError << "Invalid InnerMatrixFormat: " << format << endl << flush;
            std::exit(1);
    }
}

Foam::dfMatrix::dfMatrix(const lduMatrix& courseLduMatrix, const labelList& fineRowBlockPtr, const labelList& fineToCoarse):lduMatrix_(courseLduMatrix)
{
    // innerMatrixPtr_ = new dfLduMatrix(ldu);
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfLduMatrix>(courseLduMatrix);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfCSRMatrix>(courseLduMatrix);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR:
            Info << "Building Block CSR matrix" << endl;
            innerMatrixPtr_ = std::make_unique<dfBlockMatrix>(courseLduMatrix, fineRowBlockPtr, fineToCoarse);
            break;
        default:
            // error:
            SeriousError << "Invalid InnerMatrixFormat: " << format << endl << flush;
            std::exit(1);
    }
}

// Foam::scalarField& Foam::dfMatrix::lower()
// {
//     if (!lowerPtr_)
//     {
//         if (upperPtr_)
//         {
//             lowerPtr_ = new scalarField(*upperPtr_);
//         }
//         else
//         {
//             lowerPtr_ = new scalarField(lduAddr().lowerAddr().size(), 0.0);
//         }
//     }

//     return *lowerPtr_;
// }


// Foam::scalarField& Foam::dfMatrix::diag()
// {
//     if (!diagPtr_)
//     {
//         diagPtr_ = new scalarField(lduAddr().size(), 0.0);
//     }

//     return *diagPtr_;
// }


// Foam::scalarField& Foam::dfMatrix::upper()
// {
//     if (!upperPtr_)
//     {
//         if (lowerPtr_)
//         {
//             upperPtr_ = new scalarField(*lowerPtr_);
//         }
//         else
//         {
//             upperPtr_ = new scalarField(lduAddr().lowerAddr().size(), 0.0);
//         }
//     }

//     return *upperPtr_;
// }


// Foam::scalarField& Foam::dfMatrix::lower(const label nCoeffs)
// {
//     if (!lowerPtr_)
//     {
//         if (upperPtr_)
//         {
//             lowerPtr_ = new scalarField(*upperPtr_);
//         }
//         else
//         {
//             lowerPtr_ = new scalarField(nCoeffs, 0.0);
//         }
//     }

//     return *lowerPtr_;
// }


// Foam::scalarField& Foam::dfMatrix::diag(const label size)
// {
//     if (!diagPtr_)
//     {
//         diagPtr_ = new scalarField(size, 0.0);
//     }

//     return *diagPtr_;
// }


// Foam::scalarField& Foam::dfMatrix::upper(const label nCoeffs)
// {
//     if (!upperPtr_)
//     {
//         if (lowerPtr_)
//         {
//             upperPtr_ = new scalarField(*lowerPtr_);
//         }
//         else
//         {
//             upperPtr_ = new scalarField(nCoeffs, 0.0);
//         }
//     }

//     return *upperPtr_;
// }


// const Foam::scalarField& Foam::dfMatrix::lower() const
// {
//     if (!lowerPtr_ && !upperPtr_)
//     {
//         FatalErrorInFunction
//             << "lowerPtr_ or upperPtr_ unallocated"
//             << abort(FatalError);
//     }

//     if (lowerPtr_)
//     {
//         return *lowerPtr_;
//     }
//     else
//     {
//         return *upperPtr_;
//     }
// }


// const Foam::scalarField& Foam::dfMatrix::diag() const
// {
//     if (!diagPtr_)
//     {
//         FatalErrorInFunction
//             << "diagPtr_ unallocated"
//             << abort(FatalError);
//     }

//     return *diagPtr_;
// }


// const Foam::scalarField& Foam::dfMatrix::upper() const
// {
//     if (!lowerPtr_ && !upperPtr_)
//     {
//         FatalErrorInFunction
//             << "lowerPtr_ or upperPtr_ unallocated"
//             << abort(FatalError);
//     }

//     if (upperPtr_)
//     {
//         return *upperPtr_;
//     }
//     else
//     {
//         return *lowerPtr_;
//     }
// }
