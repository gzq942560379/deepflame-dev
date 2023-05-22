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
// #define OPT_GenMatrix_Y

#ifdef _CSR_
#include "csrMatrix.H"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif
    #include "postProcess.H"

    // #include "setRootCaseLists.H"
    #include "listOptions.H"
    #include "setRootCase2.H"

    env_show();

    double init_start = MPI_Wtime();

    double listOutput_start = MPI_Wtime();
    #include "listOutput.H"
    double listOutput_end = MPI_Wtime();

    double createTime_start = MPI_Wtime();
    #include "createTime.H"
    double createTime_end = MPI_Wtime();

    // #include "createMesh.H"
    double createDynamicFvMesh_start = MPI_Wtime();
    #include "createDynamicFvMesh.H"
    double createDynamicFvMesh_end = MPI_Wtime();

    double createDyMControls_start = MPI_Wtime();
    #include "createDyMControls.H"
    double createDyMControls_end = MPI_Wtime();

    double initContinuityErrs_start = MPI_Wtime();
    #include "initContinuityErrs.H"
    double initContinuityErrs_end = MPI_Wtime();

    double createFields_start = MPI_Wtime();
    #include "createFields.H"
    double createFields_end = MPI_Wtime();

    double createRhoUfIfPresent_start = MPI_Wtime();
    #include "createRhoUfIfPresent.H"
    double createRhoUfIfPresent_end = MPI_Wtime();

    double init_end = MPI_Wtime();

    Info << "Init time : " << init_end - init_start << endl;
    Info << "listOutput time : " << listOutput_end - listOutput_start << endl;
    Info << "createTime time : " << createTime_end - createTime_start << endl;
    Info << "createDynamicFvMesh time : " << createDynamicFvMesh_end - createDynamicFvMesh_start << endl;
    Info << "createDyMControls time : " << createDyMControls_end - createDyMControls_start << endl;
    Info << "initContinuityErrs time : " << initContinuityErrs_end - initContinuityErrs_start << endl;
    Info << "createFields time : " << createFields_end - createFields_start << endl;
    Info << "createRhoUfIfPresent time : " << createRhoUfIfPresent_end - createRhoUfIfPresent_start << endl;
    Info << endl;

    label timeIndex = 0;

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    double refine_start = 0., refine_end = 0.;
    double intializeFields_start = 0., intializeFields_end = 0.;

    while (runTime.run()){
        refine_start = MPI_Wtime();
        while (refineLevel)
        {
            double refine_one_step_start = MPI_Wtime();
            runTime++;
            #include "Refine.H"
            double refine_one_step_end = MPI_Wtime();
            Info << "refine one step time : " << refine_one_step_end - refine_one_step_start << endl; 
        }
        refine_end = MPI_Wtime();

        intializeFields_start = MPI_Wtime();
        #include "intializeFields.H"
        intializeFields_end = MPI_Wtime();

        break;
    }

    double refine_time = refine_end - refine_start;
    double intializeFields_time = intializeFields_end - intializeFields_start;

    Info << "refine time : " << refine_end - refine_start << endl; 
    Info << "intializeFields time : " << intializeFields_end - intializeFields_start << endl; 

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef _CSR_
    csrMatrix csr(mesh);
    csr.analyze();
#endif

    double total_start = MPI_Wtime();

    bool first = true;
    double first_iter_start, first_iter_end;

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        if(first){
            first_iter_start = MPI_Wtime();
        }
        
        timeIndex ++;

        double time_monitor_chem=0;
        double time_monitor_Y=0;
        double time_monitor_U=0;
        double time_monitor_p=0;
        double time_monitor_E=0;
        double time_monitor_corrThermo=0;
        double time_monitor_corrDiff=0;
        double start, end;

        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }
        
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
                // Store momentum to set rhoUf for introduced faces.
                autoPtr<volVectorField> rhoU;
                if (rhoUf.valid())
                {
                    rhoU = new volVectorField("rhoU", rho*U);
                }
                Info<< "new rhoU end" << nl << endl;
            }

            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                Info<< "rhoEqn.H start" << nl << endl;
                #include "rhoEqn.H"
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

            Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            start = MPI_Wtime();
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
                    Info<< "pEqn.H start" << nl << endl;
                    #include "pEqn.H"
                    Info<< "pEqn.H end" << nl << endl;
                }
            }
            end = MPI_Wtime();
            time_monitor_p += end - start;

            if (pimple.turbCorr())
            {
                Info<< "turbulence->correct start" << nl << endl;
                turbulence->correct();
                Info<< "turbulence->correct end" << nl << endl;
            }
        }

        rho = thermo.rho();

        runTime.write();

        Info << "output time index " << runTime.timeIndex() << endl;

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "Pimple loop count          = " << pimple_loop_count << endl;
        Info<< "Chemical sources           = " << time_monitor_chem << " s" << endl;
        Info<< "Species Equations          = " << time_monitor_Y << " s" << endl;
        Info<< "U Equations                = " << time_monitor_U << " s" << endl;
        Info<< "p Equations                = " << time_monitor_p << " s" << endl;
        Info<< "Energy Equations           = " << time_monitor_E << " s" << endl;
        Info<< "thermo & Trans Properties  = " << time_monitor_corrThermo << " s" << endl;
        Info<< "Diffusion Correction Time  = " << time_monitor_corrDiff << " s" << endl;
        Info<< "sum Time                   = " << (time_monitor_chem + time_monitor_Y + time_monitor_U + time_monitor_p + time_monitor_E + time_monitor_corrThermo + time_monitor_corrDiff) << " s" << endl;
        Info<< "============================================"<<nl<< endl;

        // Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //     << "  ClockTime = " << runTime.elapsedClockTime() << " s" << endl;
#ifdef USE_PYTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
            << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
            << "    getDNNinputsTime = " << chemistry->time_getDNNinputs() << " s"
            << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl
            << "    vec2ndarrayTime = " << chemistry->time_vec2ndarray() << " s"
            << "    pythonTime = " << chemistry->time_python() << " s"<< nl << endl;
        }
#endif
#ifdef USE_LIBTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
            << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
            << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl;
        }
#endif
        if(first){
            first_iter_end = MPI_Wtime();
            Info << "first Iter time : " << first_iter_end - first_iter_start << endl;
            first = false;
        }

    }
    double total_end = MPI_Wtime();

    double first_iter_time = first_iter_end - first_iter_start;
    double total_time = total_end - total_start;
    Info << "Total time : " << total_time - refine_time - intializeFields_time - first_iter_time << endl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
