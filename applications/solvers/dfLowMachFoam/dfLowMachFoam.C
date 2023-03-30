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
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "basicThermo.H"
#include "CombustionModel.H"

#include "csrMatrix.H"

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
    #include "listOutput.H"

    #include "createTime.H"
    #include "createMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createRhoUfIfPresent.H"

    double total_start = MPI_Wtime();

    label timeIndex = 0;

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    csrMatrix csr(mesh);
    // csr.write_pattern("sparse_pattern");

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
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
        // if (log_ && torch_)
        // {
        //     Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
        //     << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
        //     << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
        //     << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
        //     << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
        //     << "    getDNNinputsTime = " << chemistry->time_getDNNinputs() << " s"
        //     << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
        //     << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl
        //     << "    vec2ndarrayTime = " << chemistry->time_vec2ndarray() << " s"
        //     << "    pythonTime = " << chemistry->time_python() << " s"<< nl << endl;
        // }
        if (log_ && torch_)
        {
            Info
            << "    allsolveTime = " << chemistry->time_allsolve() << " s" << endl
            << "    cvodeTime = " << chemistry->time_cvode() << " s" << endl
            << "    getDNNinputsTime = " << chemistry->time_getDNNinputs() << " s" << endl
            << "    vec2ndarrayTime = " << chemistry->time_vec2ndarray() << " s" << endl
            << "    ImportPythonModuleTime = " << chemistry->time_python_import_module() << " s" << endl
            << "    PyTorchinferenceTime = " << chemistry->time_python_inference() << " s" << endl
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << endl
            << "    updateRRTime = " << chemistry->time_update_RR() << " s" << endl;
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
    }
    double total_end = MPI_Wtime();
    Info << "Total time : " << total_end - total_start << endl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
