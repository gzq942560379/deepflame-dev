#include "ellMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>
#include "env.H"

namespace Foam{

defineTypeNameAndDebug(ellMatrix, 1);


ellMatrix::ellMatrix(const lduMatrix& ldu):lduMatrix_(ldu),row_block_bit_(ell_row_block_bit), row_block_size_(1 << row_block_bit_){
    Info << "ellMatrix::ellMatrix start" << endl;
    diagonal_ = ldu.diagonal();
    symmetric_ = ldu.symmetric();
    asymmetric_ = ldu.asymmetric();
    Info << "ellMatrix::ellMatrix  0" << endl;

    const auto& lduDiag = ldu.diag();
    const auto& lduLower = ldu.lower();
    const auto& lduUpper = ldu.upper();
    const auto& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu.lduAddr().upperAddr();
    Info << "ellMatrix::ellMatrix  1" << endl;

    row_ = lduDiag.size();
    block_count_ = BLOCK_COUNT;
    diag_value_.resize(row_);
    Info << "row : " << row_ << endl;

    Info << "ellMatrix::ellMatrix  2" << endl;
    // fill diag value
    for(label i = 0; i < row_; ++i){
        diag_value_[i] = lduDiag[i];
    }
    Info << "ellMatrix::ellMatrix  3" << endl;

    // compute off_diag_count
    off_diag_count_.resize(row_);
    for(label i = 0; i < row_; ++i){
        off_diag_count_[i] = 0;
    }
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        off_diag_count_[row] += 1;
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        off_diag_count_[row] += 1;
    }
    Info << "ellMatrix::ellMatrix  4" << endl;
    
    max_count_ = 0;
    for(label i = 0; i < row_; ++i){
        max_count_ = std::max(max_count_, off_diag_count_[i]);
    }
    Info << "ellMatrix::ellMatrix  5" << endl;
    Info << "max_count_ : " << max_count_ << endl;

    // fill non-zero value
    off_diag_value_.resize(row_ * max_count_);
    off_diag_colidx_.resize(row_ * max_count_);
    for(label rbs = 0; rbs < row_; rbs += row_block_size_){
        label rbe = BLOCK_END(rbs);
        label rbl = BLOCK_LEN(rbs,rbe);
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            for(label br = 0; br < rbl; ++br){
                label row = rbs + br;
                label index = index_ellcol_start + br;
                off_diag_value_[index] = 0.;
                off_diag_colidx_[index] = row;
            }
        }
    }
    Info << "ellMatrix::ellMatrix  6" << endl;

    std::vector<label> off_diag_ellcol(row_, 0);
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        label ellcol = off_diag_ellcol[row];
        assert(ellcol < max_count_);
        label index = ELL_INDEX(row, ellcol);
        off_diag_colidx_[index] = lduLowerAddr[i];
        off_diag_value_[index] = lduLower[i];
        off_diag_ellcol[row] += 1;
    }
    Info << "ellMatrix::ellMatrix  7" << endl;
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        label ellcol = off_diag_ellcol[row];
        assert(ellcol < max_count_);
        label index = ELL_INDEX(row, ellcol);
        off_diag_colidx_[index] = lduUpperAddr[i];
        off_diag_value_[index] = lduUpper[i];
        off_diag_ellcol[row] += 1;
    }

    Info << "ellMatrix::ellMatrix 8" << endl;
    analyze();
    Info << "ellMatrix::ellMatrix end" << endl;
    // coloring();
}

void ellMatrix::write_pattern(const std::string& filename) const {
    int mpisize, mpirank;
    MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    if(mpirank == 0){
        std::stringstream ss;
        ss << filename << "_" << mpirank << ".mtx";
        FILE* fr = fopen(ss.str().c_str(),"r");
        if(fr != NULL){
            fclose(fr);
            return;
        }
        FILE* fw = fopen(ss.str().c_str(),"w");
        fprintf(fw,"%s\n", "%%MatrixMarket matrix coordinate pattern general");
        
        fprintf(fw, "%ld %ld %ld\n", row_, row_, nnz());
        for(label rbs = 0; rbs < row_; rbs += row_block_size_){
            label rbe = BLOCK_END(rbs);
            label rbl = BLOCK_LEN(rbs,rbe);
            for(label row = rbs; row < rbe; ++row){
                fprintf(fw, "%ld %ld\n", row + 1, row + 1);
            }
            label index_block_start = ELL_INDEX_BLOCK_START(rbs);
            for(label ellcol = 0; ellcol < max_count_; ++ellcol){
                label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
                for(label br = 0; br < rbl; ++br){
                    label row = rbs + br;
                    label index = index_ellcol_start + br;
                    scalar value = off_diag_value_[index];
                    label col = off_diag_colidx_[index];
                    if(col == row && value == 0.){
                        continue;
                    }
                    fprintf(fw, "%ld %ld\n", row + 1, col + 1);
                }
            }
        }

        fclose(fw);
    }
}

void ellMatrix::analyze() const {

    std::vector<label> col_count_frequency(max_count_+1,0);

    for(label r = 0; r < row_; ++r){
        label nnz_ = off_diag_count_[r];
        col_count_frequency[nnz_] += 1;
    }

    Info << "ell sparse matrix analyze : --------------------------------------------------------------------" << endl;
    Info << "row(col) : " << row_ << endl;
    Info << "block tail : " << BLOCK_TAIL << endl;
    Info << "nnz : " << nnz() << endl;
    Info << "max colume count : " << max_count_ << endl;
    Info << "fill-in : " << (((max_count_ + 1) * row_ - nnz()) * 100.0) / ((max_count_ + 1) * row_) << "%" << endl;
    Info << "sparsity : " << max_count_ * row_ * 1.0 / row_ / row_ * 100  << "%" << endl;
    Info << "colume frequency : " << endl;
    for(size_t i = 0; i < col_count_frequency.size(); ++i){
        if(col_count_frequency[i] > 0){
            Info << "\t" << i << " : " << col_count_frequency[i] << endl;
        }
    }
    Info << "------------------------------------------------------------------------------------------------" << endl;

}

void ellMatrix::check() const {

    const auto& lduDiag = ldu().diag();
    const auto& lduLower = ldu().lower();
    const auto& lduUpper = ldu().upper();
    const auto& lduLowerAddr = ldu().lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu().lduAddr().upperAddr();

    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        label col = lduLowerAddr[i];
        scalar value = lduLower[i];
        bool found = false;
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index = ELL_INDEX(row, ellcol);
            if(off_diag_colidx_[index] == col && value == off_diag_value_[index]){
                found = true;
                break;
            }
        }
        if(!found){
            assert(false);
        }
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        label col = lduUpperAddr[i];
        scalar value = lduUpper[i];
        bool found = false;
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index = ELL_INDEX(row, ellcol);
            if(off_diag_colidx_[index] == col && value == off_diag_value_[index]){
                found = true;
                break;
            }
        }
        if(!found){
            assert(false);
        }
    }
    for(label rbs = 0; rbs < row_; rbs += row_block_size_){
        label rbe = BLOCK_END(rbs);
        label rbl = BLOCK_LEN(rbs,rbe);
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            for(label br = 0; br < rbl; ++br){
                label row = rbs + br;
                label index = index_ellcol_start + br;
                scalar value = off_diag_value_[index];
                label col = off_diag_colidx_[index];
                if(ellcol >= off_diag_count_[row]){
                    assert(row == col);
                    assert(value == 0.);
                }
            }
        }
    }

}




}
