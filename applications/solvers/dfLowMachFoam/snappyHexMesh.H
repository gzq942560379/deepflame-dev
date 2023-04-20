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

Application
    snappyHexMesh

Description
    Automatic split hex mesher. Refines and snaps to surface.

\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


Info << "begin snappyMesh" << endl;
#include "addOverwriteOption.H"
Foam::argList::addBoolOption
(
    "checkGeometry",
    "check all surface geometry for quality"
);
Foam::argList::addOption
(
    "surfaceSimplify",
    "boundBox",
    "simplify the surface using snappyHexMesh starting from a boundBox"
);
Foam::argList::addOption
(
    "patches",
    "(patch0 .. patchN)",
    "only triangulate selected patches (wildcards supported)"
);
Foam::argList::addOption
(
    "outFile",
    "file",
    "name of the file to save the simplified surface to"
);
#include "addDictOption.H"

runTime.functionObjects().off();

// const bool overwrite = args.optionFound("overwrite");
const bool overwrite = true;
const bool checkGeometry = args.optionFound("checkGeometry");
const bool surfaceSimplify = args.optionFound("surfaceSimplify");

autoPtr<fvMesh> meshPtr;

{
    Foam::Info
        << "Create mesh for time = "
        << runTime.timeName() << Foam::nl << Foam::endl;

    meshPtr.set
    (
        new fvMesh
        (
            Foam::IOobject
            (
                Foam::fvMesh::defaultRegion,
                runTime.timeName(),
                runTime,
                Foam::IOobject::MUST_READ
            )
        )
    );
}

fvMesh& mesh = meshPtr();

Info<< "Read mesh in = "
    << runTime.cpuTimeIncrement() << " s" << endl;

// Check patches and faceZones are synchronised
mesh.boundaryMesh().checkParallelSync(true);
meshRefinement::checkCoupledFaceZones(mesh);


// Read meshing dictionary
const word dictName("snappyHexMeshDict");
#include "setSystemMeshDictionaryIO.H"
const IOdictionary meshDict(dictIO);


// all surface geometry
const dictionary& geometryDict = meshDict.subDict("geometry");

// refinement parameters
const dictionary& refineDict = meshDict.subDict("castellatedMeshControls");

// mesh motion and mesh quality parameters
const dictionary& motionDict = meshDict.subDict("meshQualityControls");

// snap-to-surface parameters
const dictionary& snapDict = meshDict.subDict("snapControls");

// layer addition parameters
const dictionary& layerDict = meshDict.subDict("addLayersControls");

// absolute merge distance
const scalar mergeDist = getMergeDistance
(
    mesh,
    readScalar(meshDict.lookup("mergeTolerance"))
);

const Switch keepPatches(meshDict.lookupOrDefault("keepPatches", false));



// Read decomposePar dictionary
dictionary decomposeDict;
{
    if (Pstream::parRun())
    {
        decomposeDict = IOdictionary
        (
            IOobject
            (
                "decomposeParDict",
                runTime.system(),
                mesh,
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE
            )
        );
    }
    else
    {
        decomposeDict.add("method", "none");
        decomposeDict.add("numberOfSubdomains", 1);
    }
}


// Debug
// ~~~~~

// Set debug level
meshRefinement::debugType debugLevel = meshRefinement::debugType
(
    meshDict.lookupOrDefault<label>
    (
        "debug",
        0
    )
);
{
    wordList flags;
    if (meshDict.readIfPresent("debugFlags", flags))
    {
        debugLevel = meshRefinement::debugType
        (
            meshRefinement::readFlags
            (
                meshRefinement::IOdebugTypeNames,
                flags
            )
        );
    }
}
if (debugLevel > 0)
{
    meshRefinement::debug   = debugLevel;
    snappyRefineDriver::debug = debugLevel;
    snappySnapDriver::debug   = debugLevel;
    snappyLayerDriver::debug  = debugLevel;
}

// Set file writing level
{
    wordList flags;
    if (meshDict.readIfPresent("writeFlags", flags))
    {
        meshRefinement::writeLevel
        (
            meshRefinement::writeType
            (
                meshRefinement::readFlags
                (
                    meshRefinement::IOwriteTypeNames,
                    flags
                )
            )
        );
    }
}

// Set output level
{
    wordList flags;
    if (meshDict.readIfPresent("outputFlags", flags))
    {
        meshRefinement::outputLevel
        (
            meshRefinement::outputType
            (
                meshRefinement::readFlags
                (
                    meshRefinement::IOoutputTypeNames,
                    flags
                )
            )
        );
    }
}


// Read geometry
// ~~~~~~~~~~~~~

searchableSurfaces allGeometry
(
    IOobject
    (
        "abc",                      // dummy name
        mesh.time().constant(),     // instance
        // mesh.time().findInstance("triSurface", word::null),// instance
        "triSurface",               // local
        mesh.time(),                // registry
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    geometryDict,
    meshDict.lookupOrDefault("singleRegionName", true)
);


// Read refinement surfaces
// ~~~~~~~~~~~~~~~~~~~~~~~~

autoPtr<refinementSurfaces> surfacesPtr;

Info<< "Reading refinement surfaces." << endl;

if (surfaceSimplify)
{
    IOdictionary foamyHexMeshDict
    (
        IOobject
        (
            "foamyHexMeshDict",
            runTime.system(),
            runTime,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );

    const dictionary& conformationDict =
        foamyHexMeshDict.subDict("surfaceConformation").subDict
        (
            "geometryToConformTo"
        );

    const dictionary& motionDict =
        foamyHexMeshDict.subDict("motionControl");

    const dictionary& shapeControlDict =
        motionDict.subDict("shapeControlFunctions");

    // Calculate current ratio of hex cells v.s. wanted cell size
    const scalar defaultCellSize =
        readScalar(motionDict.lookup("defaultCellSize"));

    const scalar initialCellSize = ::pow(meshPtr().V()[0], 1.0/3.0);

    // Info<< "Wanted cell size  = " << defaultCellSize << endl;
    // Info<< "Current cell size = " << initialCellSize << endl;
    // Info<< "Fraction          = " << initialCellSize/defaultCellSize
    //    << endl;

    surfacesPtr =
        createRefinementSurfaces
        (
            allGeometry,
            conformationDict,
            shapeControlDict,
            refineDict.lookupOrDefault("gapLevelIncrement", 0),
            initialCellSize/defaultCellSize
        );
}
else
{
    surfacesPtr.set
    (
        new refinementSurfaces
        (
            allGeometry,
            refineDict.subDict("refinementSurfaces"),
            refineDict.lookupOrDefault("gapLevelIncrement", 0)
        )
    );

    Info<< "Read refinement surfaces in = "
        << mesh.time().cpuTimeIncrement() << " s" << nl << endl;
}

refinementSurfaces& surfaces = surfacesPtr();


// Checking only?

if (checkGeometry)
{
    // Extract patchInfo
    List<wordList> patchTypes(allGeometry.size());

    const PtrList<dictionary>& patchInfo = surfaces.patchInfo();
    const labelList& surfaceGeometry = surfaces.surfaces();
    forAll(surfaceGeometry, surfI)
    {
        label geomI = surfaceGeometry[surfI];
        const wordList& regNames = allGeometry.regionNames()[geomI];

        patchTypes[geomI].setSize(regNames.size());
        forAll(regNames, regionI)
        {
            label globalRegionI = surfaces.globalRegion(surfI, regionI);

            if (patchInfo.set(globalRegionI))
            {
                patchTypes[geomI][regionI] =
                    word(patchInfo[globalRegionI].lookup("type"));
            }
            else
            {
                patchTypes[geomI][regionI] = wallPolyPatch::typeName;
            }
        }
    }

    // Write some stats
    allGeometry.writeStats(patchTypes, Info);
    // Check topology
    allGeometry.checkTopology(true);
    // Check geometry
    allGeometry.checkGeometry
    (
        100.0,      // max size ratio
        1e-9,       // intersection tolerance
        autoPtr<writer<scalar>>(new vtkSetWriter<scalar>()),
        0.01,       // min triangle quality
        true
    );

    return 0;
}



// Read refinement shells
// ~~~~~~~~~~~~~~~~~~~~~~

Info<< "Reading refinement shells." << endl;
shellSurfaces shells
(
    allGeometry,
    refineDict.subDict("refinementRegions")
);
Info<< "Read refinement shells in = "
    << mesh.time().cpuTimeIncrement() << " s" << nl << endl;


Info<< "Setting refinement level of surface to be consistent"
    << " with shells." << endl;
surfaces.setMinLevelFields(shells);
Info<< "Checked shell refinement in = "
    << mesh.time().cpuTimeIncrement() << " s" << nl << endl;


// Read feature meshes
// ~~~~~~~~~~~~~~~~~~~

Info<< "Reading features." << endl;
refinementFeatures features
(
    mesh,
    refineDict.lookup("features")
);
Info<< "Read features in = "
    << mesh.time().cpuTimeIncrement() << " s" << nl << endl;



// Refinement engine
// ~~~~~~~~~~~~~~~~~

Info<< nl
    << "Determining initial surface intersections" << nl
    << "-----------------------------------------" << nl
    << endl;

// Main refinement engine
meshRefinement meshRefiner
(
    mesh,
    mergeDist,          // tolerance used in sorting coordinates
    overwrite,          // overwrite mesh files?
    surfaces,           // for surface intersection refinement
    features,           // for feature edges/point based refinement
    shells              // for volume (inside/outside) refinement
);
Info<< "Calculated surface intersections in = "
    << mesh.time().cpuTimeIncrement() << " s" << nl << endl;

// Some stats
meshRefiner.printMeshInfo(debugLevel, "Initial mesh");

meshRefiner.write
(
    meshRefinement::debugType(debugLevel&meshRefinement::OBJINTERSECTIONS),
    meshRefinement::writeType(0),
    mesh.time().path()/meshRefiner.timeName()
);


// Add all the surface regions as patches
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//- Global surface region to patch (non faceZone surface) or patches
//  (faceZone surfaces)
labelList globalToMasterPatch;
labelList globalToSlavePatch;
{
    Info<< nl
        << "Adding patches for surface regions" << nl
        << "----------------------------------" << nl
        << endl;

    // From global region number to mesh patch.
    globalToMasterPatch.setSize(surfaces.nRegions(), -1);
    globalToSlavePatch.setSize(surfaces.nRegions(), -1);

    Info<< setf(ios_base::left)
        << setw(6) << "Patch"
        << setw(20) << "Type"
        << setw(30) << "Region" << nl
        << setw(6) << "-----"
        << setw(20) << "----"
        << setw(30) << "------" << endl;

    const labelList& surfaceGeometry = surfaces.surfaces();
    const PtrList<dictionary>& surfacePatchInfo = surfaces.patchInfo();

    forAll(surfaceGeometry, surfI)
    {
        label geomI = surfaceGeometry[surfI];

        const wordList& regNames = allGeometry.regionNames()[geomI];

        Info<< surfaces.names()[surfI] << ':' << nl << nl;

        if (surfaces.surfZones()[surfI].faceZoneName().empty())
        {
            // 'Normal' surface
            forAll(regNames, i)
            {
                label globalRegionI = surfaces.globalRegion(surfI, i);

                label patchi;

                if (surfacePatchInfo.set(globalRegionI))
                {
                    patchi = meshRefiner.addMeshedPatch
                    (
                        regNames[i],
                        surfacePatchInfo[globalRegionI]
                    );
                }
                else
                {
                    dictionary patchInfo;
                    patchInfo.set("type", wallPolyPatch::typeName);

                    patchi = meshRefiner.addMeshedPatch
                    (
                        regNames[i],
                        patchInfo
                    );
                }

                Info<< setf(ios_base::left)
                    << setw(6) << patchi
                    << setw(20) << mesh.boundaryMesh()[patchi].type()
                    << setw(30) << regNames[i] << nl;

                globalToMasterPatch[globalRegionI] = patchi;
                globalToSlavePatch[globalRegionI] = patchi;
            }
        }
        else
        {
            // Zoned surface
            forAll(regNames, i)
            {
                label globalRegionI = surfaces.globalRegion(surfI, i);

                // Add master side patch
                {
                    label patchi;

                    if (surfacePatchInfo.set(globalRegionI))
                    {
                        patchi = meshRefiner.addMeshedPatch
                        (
                            regNames[i],
                            surfacePatchInfo[globalRegionI]
                        );
                    }
                    else
                    {
                        dictionary patchInfo;
                        patchInfo.set("type", wallPolyPatch::typeName);

                        patchi = meshRefiner.addMeshedPatch
                        (
                            regNames[i],
                            patchInfo
                        );
                    }

                    Info<< setf(ios_base::left)
                        << setw(6) << patchi
                        << setw(20) << mesh.boundaryMesh()[patchi].type()
                        << setw(30) << regNames[i] << nl;

                    globalToMasterPatch[globalRegionI] = patchi;
                }
                // Add slave side patch
                {
                    const word slaveName = regNames[i] + "_slave";
                    label patchi;

                    if (surfacePatchInfo.set(globalRegionI))
                    {
                        patchi = meshRefiner.addMeshedPatch
                        (
                            slaveName,
                            surfacePatchInfo[globalRegionI]
                        );
                    }
                    else
                    {
                        dictionary patchInfo;
                        patchInfo.set("type", wallPolyPatch::typeName);

                        patchi = meshRefiner.addMeshedPatch
                        (
                            slaveName,
                            patchInfo
                        );
                    }

                    Info<< setf(ios_base::left)
                        << setw(6) << patchi
                        << setw(20) << mesh.boundaryMesh()[patchi].type()
                        << setw(30) << slaveName << nl;

                    globalToSlavePatch[globalRegionI] = patchi;
                }
            }
        }

        Info<< nl;
    }
    Info<< "Added patches in = "
        << mesh.time().cpuTimeIncrement() << " s" << nl << endl;
}


// Parallel
// ~~~~~~~~

// Decomposition
autoPtr<decompositionMethod> decomposerPtr
(
    decompositionMethod::New
    (
        decomposeDict
    )
);
decompositionMethod& decomposer = decomposerPtr();

if (Pstream::parRun() && !decomposer.parallelAware())
{
    FatalErrorInFunction
        << "You have selected decomposition method "
        << decomposer.typeName
        << " which is not parallel aware." << endl
        << "Please select one that is (hierarchical, ptscotch)"
        << exit(FatalError);
}

// Mesh distribution engine (uses tolerance to reconstruct meshes)
fvMeshDistribute distributor(mesh, mergeDist);





// Now do the real work -refinement -snapping -layers
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const Switch wantRefine(meshDict.lookup("castellatedMesh"));
const Switch wantSnap(meshDict.lookup("snap"));
const Switch wantLayers(meshDict.lookup("addLayers"));

// Refinement parameters
const refinementParameters refineParams(refineDict);

// Snap parameters
const snapParameters snapParams(snapDict);

// Layer addition parameters
const layerParameters layerParams(layerDict, mesh.boundaryMesh());


// if (wantRefine)
// {
    cpuTime timer;

    snappyRefineDriver refineDriver
    (
        meshRefiner,
        decomposer,
        distributor,
        globalToMasterPatch,
        globalToSlavePatch
    );


    if (!overwrite && !debugLevel)
    {
        const_cast<Time&>(mesh.time())++;
    }


    refineDriver.doRefine
    (
        refineDict,
        refineParams,
        snapParams,
        refineParams.handleSnapProblems(),
        motionDict
    );


    if (!keepPatches && !wantSnap && !wantLayers)
    {
        removeZeroSizedPatches(mesh);
    }

    // writeMesh
    // (
    //     "Refined mesh",
    //     meshRefiner,
    //     debugLevel,
    //     meshRefinement::writeLevel()
    // );

    // Info<< "Mesh refined in = "
    //     << timer.cpuTimeIncrement() << " s." << endl;
// }

// if (wantSnap)
// {
//     cpuTime timer;

//     snappySnapDriver snapDriver
//     (
//         meshRefiner,
//         globalToMasterPatch,
//         globalToSlavePatch
//     );

//     if (!overwrite && !debugLevel)
//     {
//         const_cast<Time&>(mesh.time())++;
//     }

//     // Use the resolveFeatureAngle from the refinement parameters
//     scalar curvature = refineParams.curvature();
//     scalar planarAngle = refineParams.planarAngle();

//     snapDriver.doSnap
//     (
//         snapDict,
//         motionDict,
//         curvature,
//         planarAngle,
//         snapParams
//     );

//     if (!keepPatches && !wantLayers)
//     {
//         removeZeroSizedPatches(mesh);
//     }

//     writeMesh
//     (
//         "Snapped mesh",
//         meshRefiner,
//         debugLevel,
//         meshRefinement::writeLevel()
//     );

//     Info<< "Mesh snapped in = "
//         << timer.cpuTimeIncrement() << " s." << endl;
// }

// if (wantLayers)
// {
//     cpuTime timer;

//     snappyLayerDriver layerDriver
//     (
//         meshRefiner,
//         globalToMasterPatch,
//         globalToSlavePatch
//     );

//     // Use the maxLocalCells from the refinement parameters
//     bool preBalance = returnReduce
//     (
//         (mesh.nCells() >= refineParams.maxLocalCells()),
//         orOp<bool>()
//     );


//     if (!overwrite && !debugLevel)
//     {
//         const_cast<Time&>(mesh.time())++;
//     }

//     layerDriver.doLayers
//     (
//         layerDict,
//         motionDict,
//         layerParams,
//         preBalance,
//         decomposer,
//         distributor
//     );

//     if (!keepPatches)
//     {
//         removeZeroSizedPatches(mesh);
//     }

//     writeMesh
//     (
//         "Layer mesh",
//         meshRefiner,
//         debugLevel,
//         meshRefinement::writeLevel()
//     );

//     Info<< "Layers added in = "
//         << timer.cpuTimeIncrement() << " s." << endl;
// }


// {
//     // Check final mesh
//     Info<< "Checking final mesh ..." << endl;
//     faceSet wrongFaces(mesh, "wrongFaces", mesh.nFaces()/100);
//     motionSmoother::checkMesh(false, mesh, motionDict, wrongFaces);
//     const label nErrors = returnReduce
//     (
//         wrongFaces.size(),
//         sumOp<label>()
//     );

//     if (nErrors > 0)
//     {
//         Info<< "Finished meshing with " << nErrors << " illegal faces"
//             << " (concave, zero area or negative cell pyramid volume)"
//             << endl;
//         wrongFaces.write();
//     }
//     else
//     {
//         Info<< "Finished meshing without any errors" << endl;
//     }
// }


// if (surfaceSimplify)
// {
//     const polyBoundaryMesh& bMesh = mesh.boundaryMesh();

//     labelHashSet includePatches(bMesh.size());

//     if (args.optionFound("patches"))
//     {
//         includePatches = bMesh.patchSet
//         (
//             wordReList(args.optionLookup("patches")())
//         );
//     }
//     else
//     {
//         forAll(bMesh, patchi)
//         {
//             const polyPatch& patch = bMesh[patchi];

//             if (!isA<processorPolyPatch>(patch))
//             {
//                 includePatches.insert(patchi);
//             }
//         }
//     }

//     fileName outFileName
//     (
//         args.optionLookupOrDefault<fileName>
//         (
//             "outFile",
//             "constant/triSurface/simplifiedSurface.stl"
//         )
//     );

//     extractSurface
//     (
//         mesh,
//         runTime,
//         includePatches,
//         outFileName
//     );

//     pointIOField cellCentres
//     (
//         IOobject
//         (
//             "internalCellCentres",
//             runTime.timeName(),
//             mesh,
//             IOobject::NO_READ,
//             IOobject::AUTO_WRITE
//         ),
//         mesh.cellCentres()
//     );

//     cellCentres.write();
// }