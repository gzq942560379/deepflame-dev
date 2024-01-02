template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN_blas
(
    const DeltaTType& deltaT
)
{
    double solve_DNN_blas_start = MPI_Wtime();

    double prepare_dnn_input_start = MPI_Wtime();

    scalar deltaTMin = great;
    if (!this->chemistry_)
    {
        return deltaTMin;
    }
    Info<<"=== begin solve_DNN with BLASDNN === "<<endl;

    std::vector<size_t> Cell0, Cell1, Cell2;
    
    const double lambda = 0.1;

    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar Qdoti = Qdot_[cellI];
        
        // NN0
        if (((Qdoti < 3e7) && (Ti < 2000) && ( Ti >= 700)) || (Ti < 700))//choose1
        {
            Cell0.push_back(cellI);
            continue;
        }
        // NN1
        if(((Qdoti >= 3e7) && (Ti < 2000) && (Ti >= 700)) || ((Qdoti > 7e8) && Ti > 2000)) //choose2
        {

            Cell1.push_back(cellI);
            continue;
        }
        // NN2
        if  ((Qdoti < 7e8) && (Ti >= 2000) && (Qdoti!=0)) //choose3
        {
            Cell2.push_back(cellI);
            continue;
        }
    }

    int input_dim = mixture_.nSpecies() + 2;
    
    __fp16* NNInputs0 = (__fp16*)aligned_alloc(64, (Cell0.size() * input_dim) * sizeof(__fp16));
    __fp16* NNInputs1 = (__fp16*)aligned_alloc(64, (Cell1.size() * input_dim) * sizeof(__fp16));
    __fp16* NNInputs2 = (__fp16*)aligned_alloc(64, (Cell2.size() * input_dim) * sizeof(__fp16));

    memset(NNInputs0, '\0', (Cell0.size() * input_dim) * sizeof(__fp16));
    memset(NNInputs1, '\0', (Cell1.size() * input_dim) * sizeof(__fp16));
    memset(NNInputs2, '\0', (Cell2.size() * input_dim) * sizeof(__fp16));

    // float* NNInputs0 = (float*)aligned_alloc(64, (Cell0.size() * input_dim) * sizeof(float));
    // float* NNInputs1 = (float*)aligned_alloc(64, (Cell1.size() * input_dim) * sizeof(float));
    // float* NNInputs2 = (float*)aligned_alloc(64, (Cell2.size() * input_dim) * sizeof(float));

    // memset(NNInputs0, '\0', (Cell0.size() * input_dim) * sizeof(float));
    // memset(NNInputs1, '\0', (Cell1.size() * input_dim) * sizeof(float));
    // memset(NNInputs2, '\0', (Cell2.size() * input_dim) * sizeof(float));

    // double* NNInputs0 = (double*)aligned_alloc(64, (Cell0.size() * input_dim) * sizeof(double));
    // double* NNInputs1 = (double*)aligned_alloc(64, (Cell1.size() * input_dim) * sizeof(double));
    // double* NNInputs2 = (double*)aligned_alloc(64, (Cell2.size() * input_dim) * sizeof(double));

    // memset(NNInputs0, '\0', (Cell0.size() * input_dim) * sizeof(double));
    // memset(NNInputs1, '\0', (Cell1.size() * input_dim) * sizeof(double));
    // memset(NNInputs2, '\0', (Cell2.size() * input_dim) * sizeof(double));

    #pragma omp parallel
    {
        scalarList yPre_(mixture_.nSpecies());
        scalarList yBCT_(mixture_.nSpecies());
        scalarList yTmp_(mixture_.nSpecies());
        scalarList u_(mixture_.nSpecies());
        // get problems
        #pragma omp for
        for(size_t celli = 0; celli<Cell0.size(); ++celli){
            size_t cellI = Cell0[celli];

            scalar Ti = T_[cellI];
            scalar pi = p_[cellI];
            size_t offset = celli * input_dim;
            NNInputs0[offset + 0] = (Ti - Xmu0_[0])/Xstd0_[0];
            NNInputs0[offset + 1] = (pi / 101325 - Xmu0_[1])/Xstd0_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                NNInputs0[offset + 2 + i] = (yBCT_[i] - Xmu0_[i+2]) / Xstd0_[i+2];
            }
            NNInputs0[offset + 2 + CanteraGas_->nSpecies()] = 0.; // Y_AR
        }
        #pragma omp for
        for(size_t celli = 0; celli<Cell1.size(); ++celli){
            size_t cellI = Cell1[celli];
            scalar Ti = T_[cellI];
            scalar pi = p_[cellI];
            size_t offset = celli * input_dim;
            NNInputs1[offset + 0] = (Ti - Xmu1_[0])/Xstd1_[0];
            NNInputs1[offset + 1] = (pi / 101325 - Xmu1_[1])/Xstd1_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                NNInputs1[offset + 2 + i] = (yBCT_[i] - Xmu1_[i+2]) / Xstd1_[i+2];
            }
            NNInputs1[offset + 2 + CanteraGas_->nSpecies()] = 0.; // Y_AR
        }
        #pragma omp for
        for(size_t celli = 0; celli<Cell2.size(); ++celli){
            size_t cellI = Cell2[celli];
            scalar Ti = T_[cellI];
            scalar pi = p_[cellI];
            size_t offset = celli * input_dim;
            NNInputs2[offset + 0] = (Ti - Xmu2_[0])/Xstd2_[0];
            NNInputs2[offset + 1] = (pi / 101325 - Xmu2_[1])/Xstd2_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++) // minus 1 for Y_AR
            {
                NNInputs2[offset + 2 + i] = (yBCT_[i] - Xmu2_[i+2]) / Xstd2_[i+2];
            }
            NNInputs2[offset + 2 + CanteraGas_->nSpecies()] = 0.; // Y_AR
        }
    }


    double prepare_dnn_input_end = MPI_Wtime();

    // module inference
    int output_dim = mixture_.nSpecies() + 2;

    __fp16* results0 = (__fp16*)aligned_alloc(64, (Cell0.size() * output_dim) * sizeof(__fp16));
    __fp16* results1 = (__fp16*)aligned_alloc(64, (Cell1.size() * output_dim) * sizeof(__fp16));
    __fp16* results2 = (__fp16*)aligned_alloc(64, (Cell2.size() * output_dim) * sizeof(__fp16));

    // float* results0 = (float*)aligned_alloc(64, (Cell0.size() * output_dim) * sizeof(float));
    // float* results1 = (float*)aligned_alloc(64, (Cell1.size() * output_dim) * sizeof(float));
    // float* results2 = (float*)aligned_alloc(64, (Cell2.size() * output_dim) * sizeof(float));

    // double* results0 = (double*)aligned_alloc(64, (Cell0.size() * output_dim) * sizeof(double));
    // double* results1 = (double*)aligned_alloc(64, (Cell1.size() * output_dim) * sizeof(double));
    // double* results2 = (double*)aligned_alloc(64, (Cell2.size() * output_dim) * sizeof(double));

    if(Cell0.size() + Cell1.size() + Cell2.size() < (size_t)mesh_.nCells()){
        Warning << "Cell0.size() + Cell1.size() + Cell2.size() < mesh_.nCells()" << endl;
    }

    double dnn_infer_start = MPI_Wtime();
    DNNInferencer_blas_.Inference_multiDNNs(
        NNInputs0, results0, Cell0.size(),
        NNInputs1, results1, Cell1.size(),
        NNInputs2, results2, Cell2.size());
    double dnn_infer_end = MPI_Wtime();

    double update_start = MPI_Wtime();

    // update Q & RR
    #pragma omp parallel
    {
        scalarList yPre_(mixture_.nSpecies());
        scalarList yBCT_(mixture_.nSpecies());
        scalarList yTmp_(mixture_.nSpecies());
        scalarList u_(mixture_.nSpecies());
        // - NN0
        #pragma omp for
        for(size_t cellI = 0; cellI<Cell0.size(); cellI ++)
        {
            size_t offset = cellI * (CanteraGas_->nSpecies() + 2);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yPre_[i] = Y_[i][Cell0[cellI]];
                yTmp_[i] = Y_[i][Cell0[cellI]];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function reverse-BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                u_[i] = results0[offset + i + 2]*Ystd0_[i]+Ymu0_[i];
                yTmp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,1/lambda);
                Yt += yTmp_[i]; // normalization
            }
            Qdot_[Cell0[cellI]] = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yTmp_[i] = yTmp_[i] / Yt;
                RR_[i][Cell0[cellI]] = (yTmp_[i] - Y_[i][Cell0[cellI]])*rho_[Cell0[cellI]]/1e-6;
                Qdot_[Cell0[cellI]] -= hc_[i]*RR_[i][Cell0[cellI]];
            }
        }
        // - NN1
        #pragma omp for
        for(size_t cellI = 0; cellI<Cell1.size(); cellI ++)
        {
            size_t offset = cellI * (CanteraGas_->nSpecies() + 2);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yPre_[i] = Y_[i][Cell1[cellI]];
                yTmp_[i] = Y_[i][Cell1[cellI]];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function reverse-BCT
            }
            for (size_t i=0; i<(CanteraGas_->nSpecies() - 1); i++) // minus 1 for Y_AR
            {
                u_[i] = results1[offset + i + 2]*Ystd1_[i]+Ymu1_[i];
                yTmp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,1/lambda);
                Yt += yTmp_[i];
            }
            Qdot_[Cell1[cellI]] = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yTmp_[i] = yTmp_[i] / Yt;
                RR_[i][Cell1[cellI]] = (yTmp_[i] - Y_[i][Cell1[cellI]])*rho_[Cell1[cellI]]/1e-6;
                Qdot_[Cell1[cellI]] -= hc_[i]*RR_[i][Cell1[cellI]];
            }
        }
        // - NN2
        #pragma omp for
        for(size_t cellI = 0; cellI<Cell2.size(); cellI ++)
        {
            size_t offset = cellI * (CanteraGas_->nSpecies() + 2);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yPre_[i] = Y_[i][Cell2[cellI]];
                yTmp_[i] = Y_[i][Cell2[cellI]];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function reverse-BCT
            }
            for (size_t i=0; i<(CanteraGas_->nSpecies() - 1); i++) // minus 1 for Y_AR
            {
                u_[i] = results2[offset + i + 2]*Ystd2_[i]+Ymu2_[i];
                yTmp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,1/lambda);
                Yt += yTmp_[i];
            }
            Qdot_[Cell2[cellI]] = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yTmp_[i] = yTmp_[i] / Yt;
                RR_[i][Cell2[cellI]] = (yTmp_[i] - Y_[i][Cell2[cellI]])*rho_[Cell2[cellI]]/1e-6;
                Qdot_[Cell2[cellI]] -= hc_[i]*RR_[i][Cell2[cellI]];
            }
        }
    }

    double update_end = MPI_Wtime();
    free(NNInputs0);
    free(NNInputs1);
    free(NNInputs2);
    free(results0);
    free(results1);
    free(results2);

    Info << "=== end solve_DNN with BLASDNN === " << endl;

    double solve_DNN_blas_end = MPI_Wtime();

    double solve_DNN_blas_time = solve_DNN_blas_end - solve_DNN_blas_start;
    double prepare_dnn_input_time = prepare_dnn_input_end - prepare_dnn_input_start;
    double dnn_infer_time = dnn_infer_end - dnn_infer_start;
    double update_time = update_end - update_start;

    Info << "solve_DNN_blas time : " << solve_DNN_blas_time << endl;
    Info << "prepare dnn input time : " << prepare_dnn_input_time << endl;
    Info << "dnn inference time : " << dnn_infer_time << endl;
    Info << "update time : " << update_time << endl;
    Info << "other time : " << solve_DNN_blas_time - prepare_dnn_input_time - dnn_infer_time - update_time << endl;

    return deltaTMin;
}
