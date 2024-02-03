template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::thermoDNN_blas(const scalarField& he, const scalarField& p, const scalarField& mixfrac,
                scalarField& rho, scalarField& T, scalarField& psi, scalarField& mu,
                scalarField& alpha, PtrList<volScalarField>& rhoD)
{
    double solve_DNN_blas_start = MPI_Wtime();

    double prepare_dnn_input_start = MPI_Wtime();

    label nCells = he.size();
    label input_dim = 3;
    label output_dim = 5;

    Info<<"=== begin solve_thermo with BLASDNN === "<<endl;

    float* NNInputs = (float*)aligned_alloc(64, (nCells * input_dim) * sizeof(float));
    memset(NNInputs, '\0', (nCells * input_dim) * sizeof(float));

    // Prepare input for DNN
#ifdef _OPENMP
#pragma omp parallel for
#endif
    forAll(he, cellI)
    {
        NNInputs[cellI * input_dim + 0] = ((std::pow(mixfrac[cellI], 0.1) - 1.) * 10. - thermomu_.back()) / thermostd_.back();
        NNInputs[cellI * input_dim + 1] = (he[cellI] - thermomu_[5]) / thermostd_[5];
        NNInputs[cellI * input_dim + 2] = (p[cellI] - thermomu_[6]) / thermostd_[6];
    }

    double prepare_dnn_input_end = MPI_Wtime();

    // module inference
    float* results0 = (float*)aligned_alloc(64, nCells * sizeof(float));
    float* results1 = (float*)aligned_alloc(64, nCells * sizeof(float));
    float* results2 = (float*)aligned_alloc(64, nCells * sizeof(float));
    float* results3 = (float*)aligned_alloc(64, nCells * sizeof(float));
    float* results4 = (float*)aligned_alloc(64, nCells * sizeof(float));

    double dnn_infer_start = MPI_Wtime();
    DNNThermo_blas_.Inference(nCells, NNInputs, 
        results0, results1, results2, results3, results4);
    double dnn_infer_end = MPI_Wtime();

    double update_start = MPI_Wtime();

    // update thermo properties
#ifdef _OPENMP
#pragma omp parallel for
#endif
    forAll(he, cellI){
        rho[cellI] = std::pow((results0[cellI] * thermostd_[0] + thermomu_[0]) * 0.1 + 1., 10.) * 1000;
        T[cellI] = results1[cellI] * thermostd_[1] + thermomu_[1];
        psi[cellI] = results2[cellI] * thermostd_[2] + thermomu_[2];
        mu[cellI] = results3[cellI] * thermostd_[3] + thermomu_[3];
        alpha[cellI] = results4[cellI] * thermostd_[4] + thermomu_[4];
        forAll(rhoD_, i){
            rhoD_[i][cellI] = alpha[cellI];
        }
    }
    
    double update_end = MPI_Wtime();
    free(NNInputs);
    free(results0);
    free(results1);
    free(results2);
    free(results3);
    free(results4);

    Info << "=== end solve_thermo with BLASDNN === " << endl;

    double solve_DNN_blas_end = MPI_Wtime();
    Info << "solve_DNN_blas_thermo profiling --------------------------------------------------" << endl;
    Info << "solve_DNN_blas_thermo time: " << solve_DNN_blas_end - solve_DNN_blas_start << "s" << endl;
    Info << "prepare_dnn_input time: " << prepare_dnn_input_end - prepare_dnn_input_start << "s" << endl;
    Info << "dnn_infer time: " << dnn_infer_end - dnn_infer_start << "s" << endl;
    Info << "update time: " << update_end - update_start << "s" << endl;
    Info << "solve_DNN_blas profiling --------------------------------------------------" << endl;
}