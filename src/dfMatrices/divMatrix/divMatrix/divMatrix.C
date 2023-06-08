#include "divMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>
#include <set>

namespace Foam{

defineTypeNameAndDebug(divMatrix, 0);

divMatrix::divMatrix(const lduMatrix& ldu):lduMatrix_(ldu){
    diagonal_ = ldu.diagonal();
    symmetric_ = ldu.symmetric();
    asymmetric_ = ldu.asymmetric();

    const auto& lduDiag = ldu.diag();
    const auto& lduLower = ldu.lower();
    const auto& lduUpper = ldu.upper();
    const auto& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu.lduAddr().upperAddr();

    this->row_ = lduDiag.size();
    this->diag_value_.resize(row_);
    // fill diag value
    for(label i = 0; i < row_; ++i){
        diag_value_[i] = lduDiag[i];
    }

    // // count distence size
    // std::set<label> distence_set;
    // for(label i = 0; i < lduLower.size(); ++i){
    //     label row = lduUpperAddr[i];
    //     label col = lduLowerAddr[i];
    //     label distence = col - row;
    //     Info << row << ", " << col << ", " << distence << endl;
    // }

    // for(auto it = distence_set.begin(); it != distence_set.end(); ++it){
    //     off_diag_distance_.append(*it);
    // }

    


}

void divMatrix::analyze(){

    Info << "divMatrix analyze --------------------------------------------------" << endl;
    Info << "--------------------------------------------------------------------" << endl;
    Info << "row(col) : " << row_ << endl;
    Info << "distence size : " << off_diag_distance_.size() << endl;
    Info << "distence : ";
    for(label i = 0; i < off_diag_distance_.size(); ++i){
        Info << off_diag_distance_[i] << ", ";
    }
    Info << endl;
    Info << "--------------------------------------------------------------------" << endl;
}


}
