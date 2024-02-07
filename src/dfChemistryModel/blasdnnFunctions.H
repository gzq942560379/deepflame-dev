template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN_blas
(
    const DeltaTType& deltaT
)
{
    double init_time = 0.;
    double prepare_time = 0.;
    double inference_time = 0.;
    double update_time = 0.;
    
    clockTime clock;

    scalar deltaTMin = great;

    Info<<"=== begin solve_DNN with BLASDNN === "<<endl;

    label nCells = mesh_.nCells();

    std::vector<label> Cell0, Cell1, Cell2;
    Cell0.reserve(nCells);
    Cell1.reserve(nCells);
    Cell2.reserve(nCells);
    
    const scalar lambda = 0.1;
    const scalar lambdaR = 1 / lambda;
    const scalar CONST_101325 = 101325.;
    const scalar CONSTR_101325 = 1. / CONST_101325;

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

    int nSpecies = mixture_.nSpecies();
    int input_dim = nSpecies + 2;
    int output_dim = nSpecies + 2;

    // NNInputs0 = (__fp16*)df_alloc((Cell0.size() * input_dim) * sizeof(__fp16));
    // NNInputs1 = (__fp16*)df_alloc((Cell1.size() * input_dim) * sizeof(__fp16));
    // NNInputs2 = (__fp16*)df_alloc((Cell2.size() * input_dim) * sizeof(__fp16));

    // if(input_buffer == nullptr && output_buffer == nullptr){
    //     input_buffer = (__fp16*)df_alloc((mesh_.nCells() * input_dim) * sizeof(__fp16));
    // }

    // NNInputs0 = input_buffer;
    // NNInputs1 = NNInputs0 + Cell0.size() * input_dim;
    // NNInputs2 = NNInputs1 + Cell1.size() * input_dim;

    // memset(NNInputs0, '\0', (Cell0.size() * input_dim) * sizeof(__fp16));
    // memset(NNInputs1, '\0', (Cell1.size() * input_dim) * sizeof(__fp16));
    // memset(NNInputs2, '\0', (Cell2.size() * input_dim) * sizeof(__fp16));

    // NNInputs0 = (float*)df_alloc((Cell0.size() * input_dim) * sizeof(float));
    // NNInputs1 = (float*)df_alloc((Cell1.size() * input_dim) * sizeof(float));
    // NNInputs2 = (float*)df_alloc((Cell2.size() * input_dim) * sizeof(float));

    // memset(NNInputs0, '\0', (Cell0.size() * input_dim) * sizeof(float));
    // memset(NNInputs1, '\0', (Cell1.size() * input_dim) * sizeof(float));
    // memset(NNInputs2, '\0', (Cell2.size() * input_dim) * sizeof(float));

    NNInputs0 = (double*)df_alloc((Cell0.size() * input_dim) * sizeof(double));
    NNInputs1 = (double*)df_alloc((Cell1.size() * input_dim) * sizeof(double));
    NNInputs2 = (double*)df_alloc((Cell2.size() * input_dim) * sizeof(double));

    memset(NNInputs0, '\0', (Cell0.size() * input_dim) * sizeof(double));
    memset(NNInputs1, '\0', (Cell1.size() * input_dim) * sizeof(double));
    memset(NNInputs2, '\0', (Cell2.size() * input_dim) * sizeof(double));

    // module inference

    // results0 = (__fp16*)df_alloc((Cell0.size() * output_dim) * sizeof(__fp16));
    // results1 = (__fp16*)df_alloc((Cell1.size() * output_dim) * sizeof(__fp16));
    // results2 = (__fp16*)df_alloc((Cell2.size() * output_dim) * sizeof(__fp16));

    // results0 = (float*)df_alloc((Cell0.size() * output_dim) * sizeof(float));
    // results1 = (float*)df_alloc((Cell1.size() * output_dim) * sizeof(float));
    // results2 = (float*)df_alloc((Cell2.size() * output_dim) * sizeof(float));

    results0 = (double*)df_alloc((Cell0.size() * output_dim) * sizeof(double));
    results1 = (double*)df_alloc((Cell1.size() * output_dim) * sizeof(double));
    results2 = (double*)df_alloc((Cell2.size() * output_dim) * sizeof(double));

    init_time += clock.timeIncrement();
    
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        scalarList yPre_(nSpecies);
        scalarList yBCT_(nSpecies);
        scalarList yTmp_(nSpecies);
        scalarList u_(nSpecies);
        // get problems
#ifdef _OPENMP
        #pragma omp for
#endif
        for(size_t celli = 0; celli<Cell0.size(); ++celli){
            label cellI = Cell0[celli];
            scalar Ti = T_[cellI];
            scalar pi = p_[cellI];
            size_t offset = celli * input_dim;
            NNInputs0[offset + 0] = (Ti - Xmu0_[0]) * XstdR0_[0];
            NNInputs0[offset + 1] = (pi * CONSTR_101325 - Xmu0_[1]) * XstdR0_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) * lambdaR; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                NNInputs0[offset + 2 + i] = (yBCT_[i] - Xmu0_[i+2]) * XstdR0_[i+2];
            }
            NNInputs0[offset + 2 + CanteraGas_->nSpecies()] = 0.; // Y_AR
        }
#ifdef _OPENMP
        #pragma omp for
#endif
        for(size_t celli = 0; celli<Cell1.size(); ++celli){
            label cellI = Cell1[celli];
            scalar Ti = T_[cellI];
            scalar pi = p_[cellI];
            size_t offset = celli * input_dim;
            NNInputs1[offset + 0] = (Ti - Xmu1_[0]) * XstdR1_[0];
            NNInputs1[offset + 1] = (pi * CONSTR_101325 - Xmu1_[1]) * XstdR1_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) * lambdaR; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                NNInputs1[offset + 2 + i] = (yBCT_[i] - Xmu1_[i+2]) * XstdR1_[i+2];
            }
            NNInputs1[offset + 2 + CanteraGas_->nSpecies()] = 0.; // Y_AR
        }
#ifdef _OPENMP
        #pragma omp for
#endif
        for(size_t celli = 0; celli<Cell2.size(); ++celli){
            label cellI = Cell2[celli];
            scalar Ti = T_[cellI];
            scalar pi = p_[cellI];
            size_t offset = celli * input_dim;
            NNInputs2[offset + 0] = (Ti - Xmu2_[0]) * XstdR2_[0];
            NNInputs2[offset + 1] = (pi * CONSTR_101325 - Xmu2_[1]) * XstdR2_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) * lambdaR; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++) // minus 1 for Y_AR
            {
                NNInputs2[offset + 2 + i] = (yBCT_[i] - Xmu2_[i+2]) * XstdR2_[i+2];
            }
            NNInputs2[offset + 2 + CanteraGas_->nSpecies()] = 0.; // Y_AR
        }
    }


    prepare_time += clock.timeIncrement();

    // if(Cell0.size() + Cell1.size() + Cell2.size() < (size_t)mesh_.nCells()){
    //     Warning << "Cell0.size() + Cell1.size() + Cell2.size() < mesh_.nCells()" << endl;
    // }

    DNNInferencer_blas_.Inference_multiDNNs(
        NNInputs0, results0, Cell0.size(),
        NNInputs1, results1, Cell1.size(),
        NNInputs2, results2, Cell2.size());

    inference_time += clock.timeIncrement();

    // update Q & RR
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        scalarList yPre_(nSpecies);
        scalarList yBCT_(nSpecies);
        scalarList yTmp_(nSpecies);
        scalarList u_(nSpecies);
        // - NN0
#ifdef _OPENMP
        #pragma omp for
#endif
        for(size_t cellI = 0; cellI<Cell0.size(); cellI ++)
        {
            label cellId = Cell0[cellI];
            size_t offset = cellI * (CanteraGas_->nSpecies() + 2);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yPre_[i] = Y_[i][cellId];
                yTmp_[i] = Y_[i][cellId];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) * lambdaR; // function reverse-BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                u_[i] = results0[offset + i + 2]*Ystd0_[i]+Ymu0_[i];
                yTmp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,lambdaR);
                Yt += yTmp_[i]; // normalization
            }
            Qdot_[cellId] = 0;
            scalar YtR = 1. / Yt;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yTmp_[i] = yTmp_[i] * YtR;
                RR_[i][cellId] = (yTmp_[i] - Y_[i][cellId])*rho_[cellId] * 1e6;
                Qdot_[cellId] -= hc_[i]*RR_[i][cellId];
            }
        }
        // - NN1
#ifdef _OPENMP
        #pragma omp for
#endif
        for(size_t cellI = 0; cellI<Cell1.size(); cellI ++)
        {
            label cellId = Cell1[cellI];
            size_t offset = cellI * (CanteraGas_->nSpecies() + 2);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yPre_[i] = Y_[i][cellId];
                yTmp_[i] = Y_[i][cellId];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) * lambdaR; // function reverse-BCT
            }
            for (size_t i=0; i<(CanteraGas_->nSpecies() - 1); i++) // minus 1 for Y_AR
            {
                u_[i] = results1[offset + i + 2]*Ystd1_[i]+Ymu1_[i];
                yTmp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,lambdaR);
                Yt += yTmp_[i];
            }
            Qdot_[cellId] = 0;
            scalar YtR = 1. / Yt;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yTmp_[i] = yTmp_[i] * YtR;
                RR_[i][cellId] = (yTmp_[i] - Y_[i][cellId])*rho_[cellId] * 1e6;
                Qdot_[cellId] -= hc_[i]*RR_[i][cellId];
            }
        }
        // - NN2
#ifdef _OPENMP
        #pragma omp for
#endif
        for(size_t cellI = 0; cellI<Cell2.size(); cellI ++)
        {
            label cellId = Cell2[cellI];
            size_t offset = cellI * (CanteraGas_->nSpecies() + 2);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yPre_[i] = Y_[i][cellId];
                yTmp_[i] = Y_[i][cellId];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) * lambdaR; // function reverse-BCT
            }
            for (size_t i=0; i<(CanteraGas_->nSpecies() - 1); i++) // minus 1 for Y_AR
            {
                u_[i] = results2[offset + i + 2] * Ystd2_[i] + Ymu2_[i];
                yTmp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,lambdaR);
                Yt += yTmp_[i];
            }
            Qdot_[cellId] = 0;
            scalar YtR = 1. / Yt;
            for (size_t i=0; i<CanteraGas_->nSpecies() - 1; i++) // minus 1 for Y_AR
            {
                yTmp_[i] = yTmp_[i] * YtR;
                RR_[i][cellId] = (yTmp_[i] - Y_[i][cellId]) * rho_[cellId] * 1e6;
                Qdot_[cellId] -= hc_[i]*RR_[i][cellId];
            }
        }
    }

    update_time = clock.timeIncrement();

    df_free(NNInputs0);
    df_free(NNInputs1);
    df_free(NNInputs2);
    df_free(results0);
    df_free(results1);
    df_free(results2);

    Info << "=== end solve_DNN with BLASDNN === " << endl;
    
    double solve_DNN_blas_time = clock.elapsedTime();

    Info << "solve_DNN_blas profiling --------------------------------------------------" << endl;
    Info << "solve_DNN_blas time : " << solve_DNN_blas_time << endl;
    Info << "init time : " << init_time << endl;
    Info << "prepare time : " << prepare_time << endl;
    Info << "inference time : " << inference_time << endl;
    Info << "update time : " << update_time << endl;
    Info << "other time : " << solve_DNN_blas_time - init_time - prepare_time - inference_time - update_time << endl;
    Info << "---------------------------------------------------------------------------" << endl;

    if (!this->chemistry_)
    {
        return deltaTMin;
    }
    return deltaTMin;
}
