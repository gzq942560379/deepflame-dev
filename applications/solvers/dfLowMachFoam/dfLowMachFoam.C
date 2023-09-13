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
    rhoPimpleFoam

Description
    Transient solver for turbulent flow of compressible fluids for HVAC and
    similar applications, with optional mesh motion and mesh topology changes.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient simulations.

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "CanteraMixture.H"
#include "hePsiThermo.H"

#ifdef USE_PYTORCH
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> //used to convert
#endif

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include "DNNInferencer.H"
#endif

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "basicThermo.H"
#include "CombustionModel.H"
#include "CorrectPhi.H"

#include <typeinfo>
#include "env.H"
#include "GenFvMatrix.H"

// #define _CSR_
// #define _ELL_
// #define _DIV_
// #define _LDU_
// #define OPT_GenMatrix_Y
// #define OPT_GenMatrix_E
// #define OPT_GenMatrix_U
// #define OPT_GenMatrix_p
// #define OPT_GenMatrix_U_check
// #define OPT_GenMatrix_Y_check
// #define OPT_GenMatrix_E_check
// #define OPT_GenMatrix_p_check

#ifdef _CSR_
#include "csrMatrix.H"
#endif
#ifdef _ELL_
#include "ellMatrix.H"
#endif
#ifdef _DIV_
#include "divMatrix.H"
#endif
#ifdef _LDU_
#include "LDUMatrix.H"
#endif
#include <assert.h>

// renumber
#include "argList.H"
#include "IOobjectList.H"
#include "fvMesh.H"
#include "polyTopoChange.H"
#include "ReadFields.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "SortableList.H"
#include "decompositionMethod.H"
#include "renumberMethod.H"
#include "zeroGradientFvPatchFields.H"
#include "CuthillMcKeeRenumber.H"
#include "geomRenumber.H"
#include "fvMeshSubset.H"
#include "cellSet.H"
#include "faceSet.H"
#include "pointSet.H"
#include <omp.h>

#ifdef FOAM_USE_ZOLTAN
    #include "zoltanRenumber.H"
#endif

#include "renumberMeshFuncs.H"

#define TIME
#ifdef TIME 
#define TICK0(prefix)\
    double prefix##_tick_0 = MPI_Wtime();

#define TICK(prefix,start,end)\
    double prefix##_tick_##end = MPI_Wtime();\
    Info << #prefix << "_time_" << #end << " : " << prefix##_tick_##end - prefix##_tick_##start << endl;

#else
#define TICK0(prefix) ;
#define TICK(prefix,start,end) ;
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif

    double postProcess_start = omp_get_wtime();
    #include "postProcess.H"
    double postProcess_end = omp_get_wtime();
    double postProcess_time = postProcess_end - postProcess_start;

    // #include "setRootCaseLists.H"
    double listOptions_start = omp_get_wtime();
    #include "listOptions.H"
    double listOptions_end = omp_get_wtime();
    double listOptions_time = listOptions_end - listOptions_start;

    // MPI init here
    double setRootCase2_start = omp_get_wtime();
    #include "setRootCase2.H"
    double setRootCase2_end = omp_get_wtime();
    double setRootCase2_time = setRootCase2_end - setRootCase2_start;
    
    Info << "postProcess time : " << postProcess_time << endl;
    Info << "listOptions time : " << listOptions_time << endl;
    Info << "setRootCase2 time : " << setRootCase2_time << endl;

    env_show();

    double init_start = MPI_Wtime();

    double listOutput_start = MPI_Wtime();
    #include "listOutput.H"
    double listOutput_end = MPI_Wtime();
    Info << "listOutput time : " << listOutput_end - listOutput_start << endl;

    double createTime_start = MPI_Wtime();
    #include "createTime.H"
    double createTime_end = MPI_Wtime();
    Info << "createTime time : " << createTime_end - createTime_start << endl;

    // #include "createMesh.H"
    double createDynamicFvMesh_start = MPI_Wtime();
    #include "createDynamicFvMesh.H"
    double createDynamicFvMesh_end = MPI_Wtime();
    Info << "createDynamicFvMesh time : " << createDynamicFvMesh_end - createDynamicFvMesh_start << endl;

    double createDyMControls_start = MPI_Wtime();
    #include "createDyMControls.H"
    double createDyMControls_end = MPI_Wtime();
    Info << "createDyMControls time : " << createDyMControls_end - createDyMControls_start << endl;

    double initContinuityErrs_start = MPI_Wtime();
    #include "initContinuityErrs.H"
    double initContinuityErrs_end = MPI_Wtime();
    Info << "initContinuityErrs time : " << initContinuityErrs_end - initContinuityErrs_start << endl;

    double createFields_start = MPI_Wtime();
    #include "createFields.H"
    double createFields_end = MPI_Wtime();
    Info << "createFields time : " << createFields_end - createFields_start << endl;

    double createRhoUfIfPresent_start = MPI_Wtime();
    #include "createRhoUfIfPresent.H"
    double createRhoUfIfPresent_end = MPI_Wtime();
    Info << "createRhoUfIfPresent time : " << createRhoUfIfPresent_end - createRhoUfIfPresent_start << endl;

    label timeIndex = 0;

    double turbulenceValidate_start = MPI_Wtime();
    turbulence->validate();
    double turbulenceValidate_end = MPI_Wtime();
    Info << "turbulenceValidate time : " << turbulenceValidate_end - turbulenceValidate_start << endl;

    double setInitialDeltaT_start = MPI_Wtime();
    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }
    double setInitialDeltaT_end = MPI_Wtime();
    Info << "setInitialDeltaT time : " << setInitialDeltaT_end - setInitialDeltaT_start << endl;

    double refine_start, refine_end, refine_time = 0.;
    double intializeFields_start, intializeFields_end,intializeFields_time = 0.;
    double renumber_start, renumber_end, renumber_time = 0.;
    double structureMesh_start, structureMesh_end, structureMesh_time = 0.;

    while (runTime.run()){
        refine_start = MPI_Wtime();
        while (refineLevel)
        {
            double refine_one_step_start = MPI_Wtime();
            runTime++;
            #include "Refine.H"
            double refine_one_step_end = MPI_Wtime();
            Info << "refine once time : " << refine_one_step_end - refine_one_step_start << endl; 
        }
        refine_end = MPI_Wtime();
        refine_time += refine_end - refine_start;

        intializeFields_start = MPI_Wtime();
        #include "intializeFields.H"
        intializeFields_end = MPI_Wtime();
        intializeFields_time += intializeFields_end - intializeFields_start;

        renumber_start = MPI_Wtime();
        #include "renumberMesh.H"
        renumber_end = MPI_Wtime();
        renumber_time += renumber_end - renumber_start;

        break;
    }

    structureMesh_start = MPI_Wtime();
    #include "structureMesh.H"
    structureMesh_end = MPI_Wtime();
    structureMesh_time += structureMesh_end - structureMesh_start;

    Info << "Refine time : " << refine_time << endl; 
    Info << "IntializeFields time : " << intializeFields_time << endl; 
    Info << "Renumber time : " << renumber_time << endl; 
    Info << "structureMesh time : " << structureMesh_time << endl; 
    
    double init_end = MPI_Wtime();

    double init_time = init_end - init_start;

    Info << "Init time : " << init_time << endl;

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    std::vector<double> step_timer;

    double compute_start = MPI_Wtime();

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        double time_readDyMControls=0.;
        double time_setDeltaT=0.;
        double time_monitor_rho=0.;
        double time_monitor_U=0.;
        double time_monitor_corrDiff=0.;
        double time_monitor_chem=0.;
        double time_monitor_Y=0.;
        double time_monitor_E=0.;
        double time_monitor_corrThermo=0.;
        double time_monitor_p=0.;
        double time_turbulenceCorrect=0.;
        double start, end;

        double step_start = MPI_Wtime();

        timeIndex ++;
        
        start = MPI_Wtime();
        #include "readDyMControls.H"
        end = MPI_Wtime();
        time_readDyMControls += end - start;

        start = MPI_Wtime();
        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }
        end = MPI_Wtime();
        time_setDeltaT += end - start;
        
        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        int pimple_loop_count = 0;

        while (pimple.loop())
        {
            pimple_loop_count += 1;
            if (splitting)
            {
                Info<< "YEqn_RR.H start" << nl << endl;
                #include "YEqn_RR.H"
                Info<< "YEqn_RR.H end" << nl << endl;
            }
            
            if (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
            {
                Info<< "new rhoU start" << nl << endl;
                start = MPI_Wtime();
                // Store momentum to set rhoUf for introduced faces.
                autoPtr<volVectorField> rhoU;
                if (rhoUf.valid())
                {
                    rhoU = new volVectorField("rhoU", rho*U);
                }
                end = MPI_Wtime();
                time_monitor_rho += end - start;
                Info<< "new rhoU end" << nl << endl;
            }

            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                Info<< "rhoEqn.H start" << nl << endl;
                start = MPI_Wtime();
                #include "rhoEqn.H"
                end = MPI_Wtime();
                time_monitor_rho += end - start;
                Info<< "rhoEqn.H end" << nl << endl;
            }

            Info<< "UEqn.H start" << nl << endl;
            start = MPI_Wtime();
            #include "UEqn.H"
            end = MPI_Wtime();
            Info<< "UEqn.H end" << nl << endl;
            time_monitor_U += end - start;

            if(combModelName!="ESF" && combModelName!="flareFGM" )
            {
                Info<< "YEqn.H start" << nl << endl;
                #include "YEqn.H"
                Info<< "YEqn.H end" << nl << endl;

                Info<< "EEqn.H start" << nl << endl;
                start = MPI_Wtime();
                #include "EEqn.H"
                end = MPI_Wtime();
                Info<< "EEqn.H end" << nl << endl;
                time_monitor_E += end - start;

                Info<< "chemistry->correctThermo start" << nl << endl;
                start = MPI_Wtime();
                chemistry->correctThermo();
                end = MPI_Wtime();
                Info<< "chemistry->correctThermo end" << nl << endl;
                time_monitor_corrThermo += end - start;
            }
            else
            {
                Info<< "chemistry->correct start" << nl << endl;
                combustion->correct();
                Info<< "chemistry->correct end" << nl << endl;
            }

            // Info << "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    Info<< "pcEqn.H start" << nl << endl;
                    #include "pcEqn.H"
                    Info<< "pcEqn.H end" << nl << endl;
                }
                else
                {
                    start = MPI_Wtime();
                    Info<< "pEqn.H start" << nl << endl;
                    #include "pEqn.H"
                    Info<< "pEqn.H end" << nl << endl;
                    end = MPI_Wtime();
                    time_monitor_p += end - start;
                }
            }

            if (pimple.turbCorr())
            {
                Info<< "turbulence->correct start" << nl << endl;
                start = MPI_Wtime();
                turbulence->correct();
                end = MPI_Wtime();
                time_turbulenceCorrect += end - start;
                Info<< "turbulence->correct end" << nl << endl;
            }

        }
        start = MPI_Wtime();
        rho = thermo.rho();
        end = MPI_Wtime();
        time_monitor_rho += end - start;

        runTime.write();

        Info << "output time index " << runTime.timeIndex() << endl;

        double step_end = MPI_Wtime();
        double step_time = step_end - step_start;
        step_timer.push_back(step_time);

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "Step time                  = " << step_time << endl;
        Info<< "Pimple loop count          = " << pimple_loop_count << endl;
        Info<< "readDyMControls            = " << time_readDyMControls << " s" << endl;
        Info<< "setDeltaT                  = " << time_setDeltaT << " s" << endl;
        Info<< "rho Equations              = " << time_monitor_rho << " s" << endl;
        Info<< "U Equations                = " << time_monitor_U << " s" << endl;
        Info<< "Diffusion Correction Time  = " << time_monitor_corrDiff << " s" << endl;
        Info<< "Chemical sources           = " << time_monitor_chem << " s" << endl;
        Info<< "Species Equations          = " << time_monitor_Y << " s" << endl;
        Info<< "Energy Equations           = " << time_monitor_E << " s" << endl;
        Info<< "thermo & Trans Properties  = " << time_monitor_corrThermo << " s" << endl;
        Info<< "p Equations                = " << time_monitor_p << " s" << endl;
        Info<< "turbulence Correction      = " << time_turbulenceCorrect << " s" << endl;
        Info<< "other                      = " << step_time \
            - time_readDyMControls - time_setDeltaT - time_monitor_rho \
            - time_monitor_U - time_monitor_corrDiff - time_monitor_chem - time_monitor_Y \
            - time_monitor_E - time_monitor_corrThermo - time_monitor_p - time_turbulenceCorrect \
            << endl;
        Info<< "============================================"<<nl<< endl;
    }

    double compute_end = MPI_Wtime();
    double compute_time = compute_end - compute_start - step_timer[0] - step_timer[1];


    Info << "Init time : " << init_time << endl;
    Info << "First time : " << step_timer[0] << endl;
    Info << "Second time : " << step_timer[1] << endl;
    Info << "Compute step : " << step_timer.size() - 2 << endl;
    Info << "Compute time : " << compute_time << endl;
    Info << "Total time : " << init_time + step_timer[0] + step_timer[1] + compute_time << endl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
