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

    std::vector<size_t> Cell0;

    
    const double lambda = 0.1;

    label input_dim = mixture_.nSpecies() + 2;
    label nCell = T_.size();

    float* NNInputs0 = (float*)aligned_alloc(64, (nCell * input_dim) * sizeof(float));

    memset(NNInputs0, '\0', (nCell * input_dim) * sizeof(float));
    #pragma omp parallel
    {
        scalarList yPre_(mixture_.nSpecies());
        scalarList yBCT_(mixture_.nSpecies());
        scalarList yTmp_(mixture_.nSpecies());
        scalarList u_(mixture_.nSpecies());
        // get problems
        #pragma omp for
        for(size_t celli = 0; celli<nCell; ++celli){
            // size_t celli = Cell0[celli];
            scalar Ti = T_[celli];
            scalar pi = p_[celli];
            size_t offset = celli * input_dim;

            NNInputs0[offset + 0] = (Ti - Xmu_[0])/Xstd_[0];
            NNInputs0[offset + 1] = (pi - Xmu_[1])/Xstd_[1];
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][celli];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                NNInputs0[offset + 2 + i] = (yBCT_[i] - Xmu_[i+2]) / Xstd_[i+2];
            }
        }
    }
    
    double prepare_dnn_input_end = MPI_Wtime();

    // module inference
    int output_dim = mixture_.nSpecies() -1;

    float* results0 = (float*)aligned_alloc(64, (nCell * output_dim) * sizeof(float));

    double dnn_infer_start = MPI_Wtime();
    DNNInferencer_blas_.Inference_multiDNNs(NNInputs0, results0, nCell);

    double dnn_infer_end = MPI_Wtime();

    double update_start = MPI_Wtime();

    // update Q & RR
    #pragma omp parallel
    {
        scalarList yPre_(mixture_.nSpecies());
        scalarList yBCT_(mixture_.nSpecies());
        scalarList yTmp_(mixture_.nSpecies());
        scalarList u_(mixture_.nSpecies());
        // // - NN0
        #pragma omp for
        for(size_t celli = 0; celli<nCell; celli ++)
        {
            size_t offset = celli * (CanteraGas_->nSpecies()-1);
            scalar Yt = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies() ; i++)
            {
                yPre_[i] = Y_[i][celli];
                yTmp_[i] = Y_[i][celli];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; 
            }
            for (size_t i=0; i<CanteraGas_->nSpecies() -1; i++)
            {

                u_[i] = results0[offset + i ]*Ystd_[i]+Ymu_[i];
                yTmp_[i] = pow((u_[i]+yBCT_[i])*lambda+1,1/lambda);
                Yt += yTmp_[i]; // normalization
            }
            u_[17] = 0;
            yTmp_[17] = yPre_[17] ;
            Qdot_[celli] = 0;
            RR_[17][celli] = 0;
            for (size_t i=0; i<CanteraGas_->nSpecies()-1; i++) 
            {
                if(T_[celli]<500){
                    RR_[i][celli] = 0;
                }
                else{
                    yTmp_[i] = yTmp_[i] / Yt *(1-yTmp_[17]);
                    RR_[i][celli] = (yTmp_[i] - Y_[i][celli])*rho_[celli]/1e-8;
                    Qdot_[celli] -= hc_[i]*RR_[i][celli];
                }
            }
        }
    }

    double update_end = MPI_Wtime();
    free(NNInputs0);
    free(results0);
    Info << "=== end solve_DNN with BLASDNN === " << endl;

    double solve_DNN_blas_end = MPI_Wtime();

    double solve_DNN_blas_time = solve_DNN_blas_end - solve_DNN_blas_start;
    double prepare_dnn_input_time = prepare_dnn_input_end - prepare_dnn_input_start;
    double dnn_infer_time = dnn_infer_end - dnn_infer_start;
    double update_time = update_end - update_start;

    Info << "solve_DNN_blas profiling --------------------------------------------------" << endl;
    Info << "solve_DNN_blas time : " << solve_DNN_blas_time << endl;
    Info << "prepare dnn input time : " << prepare_dnn_input_time << endl;
    Info << "dnn inference time : " << dnn_infer_time << endl;
    Info << "update time : " << update_time << endl;
    Info << "other time : " << solve_DNN_blas_time - prepare_dnn_input_time - dnn_infer_time - update_time << endl;
    Info << "---------------------------------------------------------------------------" << endl;

    return deltaTMin;
}
