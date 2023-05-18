template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN_blas
(
    const DeltaTType& deltaT
)
{
    scalar deltaTMin = great;
    Info<<"=== begin solve_DNN with BLASDNN === "<<endl;
    std::vector<float> NNInputs0, NNInputs1, NNInputs2;
    std::vector<size_t> Cell0, Cell1, Cell2;
    scalarList yPre_(mixture_.nSpecies());
    scalarList yBCT_(mixture_.nSpecies());
    scalarList u_(mixture_.nSpecies());
    double lambda = 0.1;
    // get problems
    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];

        // NN0
        if (((Qdot_[cellI] < 3e7) && (T_[cellI] < 2000) && ( T_[cellI] >= 700)) || (T_[cellI] < 700))//choose1
        {
            NNInputs0.push_back((Ti - Xmu0_[0])/Xstd0_[0]);
            NNInputs0.push_back((pi / 101325 - Xmu0_[1])/Xstd0_[1]);
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                NNInputs0.push_back((yBCT_[i] - Xmu0_[i+2]) / Xstd0_[i+2]);
            }
            Cell0.push_back(cellI);
            continue;
        }
        // NN1
        if(((Qdot_[cellI] >= 3e7) && (T_[cellI] < 2000)&&(T_[cellI] >= 700))||((Qdot_[cellI] > 7e8) && T_[cellI] > 2000)) //choose2
        {
            NNInputs1.push_back((Ti - Xmu1_[0])/Xstd1_[0]);
            NNInputs1.push_back((pi / 101325 - Xmu1_[1])/Xstd1_[1]);
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                NNInputs1.push_back((yBCT_[i] - Xmu1_[i+2]) / Xstd1_[i+2]);
            }
            Cell1.push_back(cellI);
            continue;
        }
        // NN2
        if  ((Qdot_[cellI] < 7e8) && (T_[cellI] >= 2000) && (Qdot_[cellI]!=0)) //choose3
        {
            NNInputs2.push_back((Ti - Xmu2_[0])/Xstd2_[0]);
            NNInputs2.push_back((pi / 101325 - Xmu2_[1])/Xstd2_[1]);
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                NNInputs2.push_back((yBCT_[i] - Xmu2_[i+2]) / Xstd2_[i+2]);
            }
            Cell2.push_back(cellI);
            continue;
        }
    }

    DNNInferencer_blas_.alloc_buffer(mesh_.nCells());
    // module inference
    // std::vector<std::vector<float>> NNInputs = {NNInputs0, NNInputs1, NNInputs2};
    // auto results = DNNInferencertf_.Inference_multiDNNs(NNInputs, mixture_.nSpecies()+2);
    assert(mixture_.nSpecies() + 2 == 9);
    int input_dim = mixture_.nSpecies() + 2;
    int output_dim = mixture_.nSpecies() + 2;
    std::vector<double> results0(Cell0.size() * output_dim);
    std::vector<double> results1(Cell1.size() * output_dim);
    std::vector<double> results2(Cell2.size() * output_dim);

    // std::cout << "DNN Cell0 : " << Cell0.size() << std::endl;
    // std::cout << "DNN Cell1 : " << Cell1.size() << std::endl;
    // std::cout << "DNN Cell2 : " << Cell2.size() << std::endl;

    // std::cout << "input 0 : " << Cell0.size() << std::endl;

    // for(int i = 0 ;i < 1; ++i){
    //     for(int d = 0; d < 9; ++d){
    //         std::cout << NNInputs0[i * 9 + d] << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    DNNInferencer_blas_.Inference_multiDNNs(
        NNInputs0, results0, Cell0.size(),
        NNInputs1, results1, Cell1.size(),
        NNInputs2, results2, Cell2.size());

    // std::cout << "result 0 : " << Cell0.size() << std::endl;

    // for(int i = 0 ;i < 1; ++i){
    //     for(int d = 0; d < 9; ++d){
    //         std::cout << results0[i * 9 + d] << "\t";
    //     }
    //     std::cout << std::endl;
    // }
    
    // update Q & RR
    // - NN0
    int offset;
    for(size_t cellI = 0; cellI<Cell0.size(); cellI ++)
    {
        offset = cellI * (CanteraGas_->nSpecies() + 2);
        scalar Yt = 0;
        for (int i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yPre_[i] = Y_[i][Cell0[cellI]];
            yTemp_[i] = Y_[i][Cell0[cellI]];
            yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function reverse-BCT
        }
        for (int i=0; i<CanteraGas_->nSpecies(); i++)//
        {
            u_[i] = results0[offset + i + 2]*Ystd0_[i]+Ymu0_[i];
            yTemp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,1/lambda);
            Yt += yTemp_[i]; // normalization
        }
        Qdot_[Cell0[cellI]] = 0;
        for (int i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yTemp_[i] = yTemp_[i] / Yt;
            RR_[i][Cell0[cellI]] = (yTemp_[i] - Y_[i][Cell0[cellI]])*rho_[Cell0[cellI]]/1e-6;
            Qdot_[Cell0[cellI]] -= hc_[i]*RR_[i][Cell0[cellI]];
        }
    }
    // - NN1
    for(size_t cellI = 0; cellI<Cell1.size(); cellI ++)
    {
        offset = cellI * (CanteraGas_->nSpecies() + 2);
        scalar Yt = 0;
        for (int i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yPre_[i] = Y_[i][Cell1[cellI]];
            yTemp_[i] = Y_[i][Cell1[cellI]];
            yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function reverse-BCT
        }
        for (int i=0; i<(CanteraGas_->nSpecies()); i++)//
        {
            u_[i] = results1[offset + i + 2]*Ystd1_[i]+Ymu1_[i];
            yTemp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,1/lambda);
            Yt += yTemp_[i];
        }
        Qdot_[Cell1[cellI]] = 0;
        for (int i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yTemp_[i] = yTemp_[i] / Yt;
            RR_[i][Cell1[cellI]] = (yTemp_[i] - Y_[i][Cell1[cellI]])*rho_[Cell1[cellI]]/1e-6;
            Qdot_[Cell1[cellI]] -= hc_[i]*RR_[i][Cell1[cellI]];
        }
    }
    // - NN2
    for(size_t cellI = 0; cellI<Cell2.size(); cellI ++)
    {
        offset = cellI * (CanteraGas_->nSpecies() + 2);
        scalar Yt = 0;
        for (int i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yPre_[i] = Y_[i][Cell2[cellI]];
            yTemp_[i] = Y_[i][Cell2[cellI]];
            yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function reverse-BCT
        }
        for (int i=0; i<(CanteraGas_->nSpecies()); i++)//
        {
            u_[i] = results2[offset + i + 2]*Ystd2_[i]+Ymu2_[i];
            yTemp_[i] = pow((yBCT_[i] + u_[i]*1e-6)*lambda+1,1/lambda);
            Yt += yTemp_[i];
        }
        Qdot_[Cell2[cellI]] = 0;
        for (int i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yTemp_[i] = yTemp_[i] / Yt;
            RR_[i][Cell2[cellI]] = (yTemp_[i] - Y_[i][Cell2[cellI]])*rho_[Cell2[cellI]]/1e-6;
            Qdot_[Cell2[cellI]] -= hc_[i]*RR_[i][Cell2[cellI]];
        }
    }

    Info << "=== end solve_DNN with BLASDNN === " << endl;
    return deltaTMin;
}
