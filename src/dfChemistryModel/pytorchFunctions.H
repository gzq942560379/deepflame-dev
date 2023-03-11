template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN(const DeltaTType &deltaT)
{
    scalar deltaTMin = great;
    // set the cores slaved by a DCU
    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    double start = MPI_Wtime();
    Info << "=== begin solve_DNN === " << endl;
    if (gpu_)
    {
        Info << "now DNN inference is conducted on GPU" << endl;
    }
    else
    {
        Info << "now DNN inference is conducted on CPU" << endl;
    }


    /*=============================gather problems=============================*/
    double start10 = MPI_Wtime();
    DynamicList<GpuProblem> GPUproblemList; //single core TODO:rename it
    DynamicList<ChemistryProblem> CPUproblemList;
    getGPUProblems(deltaT, GPUproblemList, CPUproblemList);
    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    double end10 = MPI_Wtime();
    time_getProblems_ += end10-start10;

    if (gpu_)
    {
        /*==============================send problems==============================*/
        double start2 = MPI_Wtime();

        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
        if (Pstream::myProcNo() % cores_) //for slave
        {
            UOPstream send((Pstream::myProcNo()/cores_)*cores_, pBufs);// sending problem to master
            send << GPUproblemList;
        }
        pBufs.finishedSends();

        /*==============================send CVODE problems from submaster to neighbour==============================*/
        PstreamBuffers pBufs1(Pstream::commsTypes::nonBlocking);
        if (!(Pstream::myProcNo() % cores_)) // submaster
        {
            UOPstream send((Pstream::myProcNo() + 1), pBufs1);// sending CPUproblems to neighbour
            send << CPUproblemList;
        }
        pBufs1.finishedSends();
        if ((Pstream::myProcNo() % cores_) == 1) // neighbour of submaster
        {
            DynamicList<ChemistryProblem> CPUproblemList_submaster;
            UIPstream recv((Pstream::myProcNo() - 1), pBufs1);
            recv >> CPUproblemList_submaster;
            CPUproblemList.append(CPUproblemList_submaster);
        }

        /*========================================================================================================*/

        DynamicBuffer<GpuSolution> solutionBuffer;

        double end2 = MPI_Wtime();
        // std::cout << "sendProblemTime = " << end2 - start2 << std::endl;
        time_sendProblem_ += end2 - start2;

        /*=============================submaster work start=============================*/
        if (!(Pstream::myProcNo() % cores_))
        {
            double start1 = MPI_Wtime();
            double start3 = MPI_Wtime();

            label problemSize = 0; // problemSize is defined to debug
            DynamicBuffer<GpuProblem> problemBuffer(cores_);//each submaster init a local problemBuffer TODO:rename it

            /*==============================gather problems==============================*/
            problemBuffer[0] = GPUproblemList; //problemList of submaster get index 0
            problemSize += problemBuffer[0].size();

            for (label i = 1; i < cores_; i++)
            {
                UIPstream recv(i + Pstream::myProcNo(), pBufs);
                recv >> problemBuffer[i];  //recv previous send problem and append to problemList
                problemSize += problemBuffer[i].size();
            }
            if (gpulog_)
            {
                Info << "problemSize = " << problemSize << endl;
            }

            double end3 = MPI_Wtime();
            // std::cout << "RecvProblemTime = " << end3 - start3 << std::endl;
            time_RecvProblem_ += end3 - start3;

            /*==============================construct DNN inputs==============================*/
            std::vector<label> outputLength;
            std::vector<std::vector<double>> DNNinputs;     // vectors for the inference of DNN
            std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
            std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave

            double start5 = MPI_Wtime();
            getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
            double end5 = MPI_Wtime();
            // std::cout << "getDNNinputsTime = " << end5 - start5 << std::endl;
            time_getDNNinputs_ += end5 - start5;

            /*=============================inference via pybind11=============================*/
            double start7 = MPI_Wtime();
            double start8 = MPI_Wtime();

            pybind11::array_t<double> vec0 = pybind11::array_t<double>({DNNinputs[0].size()}, {8}, &DNNinputs[0][0]); // cast vector to np.array
            pybind11::array_t<double> vec1 = pybind11::array_t<double>({DNNinputs[1].size()}, {8}, &DNNinputs[1][0]);
            pybind11::array_t<double> vec2 = pybind11::array_t<double>({DNNinputs[2].size()}, {8}, &DNNinputs[2][0]);

            double end8 = MPI_Wtime();
            // std::cout << "vec2ndarrayTime = " << end8 - start8 << std::endl;
            time_vec2ndarray_ += end8 - start8;

            pybind11::module_ call_torch = pybind11::module_::import("inference"); // import python file

            double start9 = MPI_Wtime();

            pybind11::object result = call_torch.attr("inference")(vec0, vec1, vec2); // call python function
            const double* star = result.cast<pybind11::array_t<double>>().data();

            double end9 = MPI_Wtime();
            // std::cout << "pythonTime = " << end9 - start9 << std::endl;
            time_python_ += end9 - start9;

            double end7 = MPI_Wtime();
            // std::cout << "DNNinferenceTime = " << end7 - start7 << std::endl;
            time_DNNinference_ += end7 - start7;

            /*=============================construct solutions=============================*/
            double start6 = MPI_Wtime();
            std::vector<double> outputsVec0(star, star+outputLength[0] * 7); //the float number is sample_length*sample_number
            std::vector<double> outputsVec1(star+outputLength[0] * 7, star+outputLength[1] * 7);
            std::vector<double> outputsVec2(star+outputLength[1] * 7, star+outputLength[2] * 7);
            std::vector<std::vector<double>> results = {outputsVec0, outputsVec1, outputsVec2};
            updateSolutionBuffer(solutionBuffer, results, cellIDBuffer, problemCounter);
            double end6 = MPI_Wtime();
            // std::cout << "updateSolutionBufferTime = " << end6 - start6 << std::endl;
            time_updateSolutionBuffer_ += end6 - start6;

            double end1 = MPI_Wtime();
            // std::cout << "submasterTime = " << end1 - start1 << std::endl;
            time_submaster_ += end1 - start1;
        }

        /*=============================calculates RR with CVODE use DLB=============================*/
        DynamicList<ChemistrySolution> CPUSolutionList;
        if (Pstream::myProcNo() % cores_) //for slave
        {
            double start = MPI_Wtime();
            DynamicBuffer<ChemistrySolution> incomingSolutions;
            balancer_.updateState(CPUproblemList, cvodeComm);
            auto guestProblems = balancer_.balance(CPUproblemList, cvodeComm);
            auto ownProblems = balancer_.getRemaining(CPUproblemList, cvodeComm);
            auto ownSolutions = solveList(ownProblems);
            auto guestSolutions = solveBuffer(guestProblems);
            incomingSolutions = balancer_.unbalance(guestSolutions, cvodeComm);
            incomingSolutions.append(ownSolutions);
            updateReactionRates(incomingSolutions, CPUSolutionList);
            double end = MPI_Wtime();
            std::cout << "slaveTime = " << end - start << std::endl;
        }

        /*=============================send CPUSolutionList back to submaster=============================*/
        PstreamBuffers pBufs3(Pstream::commsTypes::nonBlocking);

        if ((Pstream::myProcNo() % cores_) == 1) // neighbour of submaster
        {
            UOPstream send((Pstream::myProcNo() - 1), pBufs3);
            send << CPUSolutionList;
        }
        pBufs3.finishedSends();
        if (!(Pstream::myProcNo() % cores_)) // submaster
        {
            UIPstream recv((Pstream::myProcNo() + 1), pBufs3);// resv CPUproblems from neighbour
            recv >> CPUSolutionList;
        }

        /*=============================send and recv GPUSolutions=============================*/
        double start4 = MPI_Wtime();

        DynamicList<GpuSolution> finalList;
        PstreamBuffers pBufs2(Pstream::commsTypes::nonBlocking);
        if (!(Pstream::myProcNo() % cores_)) // submaster
        {
            finalList = solutionBuffer[0];
            for (label i = 1; i < cores_; i++)
            {
                UOPstream send(i + Pstream::myProcNo(), pBufs2);
                send << solutionBuffer[i];
            }
        }
        pBufs2.finishedSends();
        if (Pstream::myProcNo() % cores_) // slavers
        {
            UIPstream recv((Pstream::myProcNo()/cores_)*cores_, pBufs2);
            recv >> finalList;
        }

        double end4 = MPI_Wtime();
        // std::cout << "SendRecvSolutionTime = " << end4 - start4 << std::endl;
        time_sendRecvSolution_ += end4 - start4;

        /*=============================update RR fields=============================*/
        for (int cellI = 0; cellI < finalList.size(); cellI++)
        {
            Qdot_[finalList[cellI].cellid] = 0;
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
                Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
            }
        }

        if (!(Pstream::myProcNo() % cores_)) // submaster
        {
            for (int cellI = 0; cellI < CPUSolutionList.size(); cellI++)
            {
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    RR_[speciID][CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].RRi[speciID];
                }
                Qdot_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].Qdoti;
                cpuTimes_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].cpuTime;
            }
        }
    }
    else
    {
        double start1 = MPI_Wtime();
        cores_ = 1;
        // solve CPU problem with cvode
        DynamicBuffer<ChemistrySolution> incomingSolutions;
        DynamicList<ChemistrySolution> CPUSolutionList;
        balancer_.updateState(CPUproblemList, cvodeComm);
        auto guestProblems = balancer_.balance(CPUproblemList, cvodeComm);
        auto ownProblems = balancer_.getRemaining(CPUproblemList, cvodeComm);
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        incomingSolutions = balancer_.unbalance(guestSolutions, cvodeComm);
        incomingSolutions.append(ownSolutions);
        updateReactionRates(incomingSolutions, CPUSolutionList);
        double end1 = MPI_Wtime();
        time_cvode_ = end1 - start1;

        // solve other problems with NN
        /*==============================construct DNN inputs==============================*/
        double start2 = MPI_Wtime();
        DynamicBuffer<GpuProblem> problemBuffer;
        DynamicBuffer<GpuSolution> solutionBuffer;
        std::vector<label> outputLength;
        std::vector<std::vector<double>> DNNinputs;     // tensors for the inference of DNN
        std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
        std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave
        problemBuffer.append(GPUproblemList);
        getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
        double end2 = MPI_Wtime();
        time_getDNNinputs_ = end2 - start2;
        /*==============================vector to ndarray==============================*/
        double start3 = MPI_Wtime();
        pybind11::array_t<double> vec0 = pybind11::array_t<double>({DNNinputs[0].size()}, {8}, &DNNinputs[0][0]); // cast vector to np.array
        pybind11::array_t<double> vec1 = pybind11::array_t<double>({DNNinputs[1].size()}, {8}, &DNNinputs[1][0]);
        pybind11::array_t<double> vec2 = pybind11::array_t<double>({DNNinputs[2].size()}, {8}, &DNNinputs[2][0]);
        double end3 = MPI_Wtime();
        time_vec2ndarray_ = end3 - start3;
        
        /*==============================import python module==============================*/
        double start4 = MPI_Wtime();
        pybind11::module_ call_torch = pybind11::module_::import("inference"); // import python file
        double end4 = MPI_Wtime();
        time_python_import_module_ = end4 - start4;
        
        /*==============================call python function==============================*/
        double start5 = MPI_Wtime();
        pybind11::object result = call_torch.attr("inference")(vec0, vec1, vec2); // call python function
        double end5 = MPI_Wtime();
        time_python_inference_ = end5 - start5;

        /*=============================construct solutions=============================*/
        double start6 = MPI_Wtime();
        const double* star = result.cast<pybind11::array_t<double>>().data();
        std::vector<double> outputsVec0(star, star+outputLength[0] * 7); //the float number is sample_length*sample_number
        std::vector<double> outputsVec1(star+outputLength[0] * 7, star+outputLength[1] * 7);
        std::vector<double> outputsVec2(star+outputLength[1] * 7, star+outputLength[2] * 7);
        std::vector<std::vector<double>> results = {outputsVec0, outputsVec1, outputsVec2};
        updateSolutionBuffer(solutionBuffer, results, cellIDBuffer, problemCounter);
        double end6 = MPI_Wtime();
        time_updateSolutionBuffer_ = end6 - start6;
        /*=============================update RR fields=============================*/
        double start7 = MPI_Wtime();
        DynamicList<GpuSolution> finalList;
        finalList = solutionBuffer[0];
        for (int cellI = 0; cellI < finalList.size(); cellI++)
        {
            Qdot_[finalList[cellI].cellid] = 0;
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
                Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
            }
        }        
        double end7 = MPI_Wtime();
        time_update_RR_ = end7 - start7;
    }

    Info << "=== end solve_DNN === " << endl;

    double end = MPI_Wtime();
    // std::cout << "allSolveTime = " << end - start << std::endl;
    time_allsolve_ = end - start;

    return deltaTMin;
}