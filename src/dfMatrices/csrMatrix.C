#include "csrMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"

namespace Foam{

csrMatrix::csrMatrix(const Foam::fvMesh& mesh):mesh_(mesh){
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    assert(owner.size() == neighbour.size());
    this->row_ = mesh.nCells();
    this->col_ = mesh.nCells();
    label face_count = neighbour.size();
    this->off_diag_nnz_ = neighbour.size() * 2;

    this->diag_value_.resize(row_);
    this->off_diag_rowptr_.resize(row_ + 1);
    this->off_diag_colidx_.resize(off_diag_nnz_);
    this->off_diag_value_.resize(off_diag_nnz_);
    this->face2lower_.resize(off_diag_nnz_);
    this->face2upper_.resize(off_diag_nnz_);

    // compute off_diag_count
    std::vector<label> off_diag_count(row_, 0);
    std::vector<label> off_diag_current_index(row_ + 1);
    for(label i = 0; i < neighbour.size(); ++i){
        auto own = owner[i];
        auto nei = neighbour[i];
        off_diag_count[own] += 1;
        off_diag_count[nei] += 1;
    }

    off_diag_rowptr_[0] = 0;
    off_diag_current_index[0] = 0;
    for(label i = 0; i < row_; ++i){
        off_diag_rowptr_[i + 1] = off_diag_rowptr_[i] + off_diag_count[i];
        off_diag_current_index[i + 1] = off_diag_rowptr_[i] + off_diag_count[i];
    }

    // compute map from face to lower
    // (nei, own)
    for(label i = 0; i < neighbour.size(); ++i){
        label r = neighbour[i];
        label c = owner[i];
        label index = off_diag_current_index[r];
        face2lower_[i] = index;
        off_diag_colidx_[index] = c;
        off_diag_current_index[r] += 1;
    }

    // (own, nei)
    for(label i = 0; i < neighbour.size(); ++i){
        label r = owner[i];
        label c = neighbour[i];
        label index = off_diag_current_index[r];
        off_diag_colidx_[index] = c;
        off_diag_current_index[r] += 1;
    }
}

csrMatrix::csrMatrix(const csrMatrix& other):mesh_(other.mesh_){
    
}

csrMatrix::~csrMatrix(){

}

void csrMatrix::write_pattern(const std::string& filename){
    int mpisize, mpirank;
    MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    std::stringstream ss;
    ss << filename << "_" << mpirank << ".mtx";
    FILE* file = fopen(ss.str().c_str(),"w");
    if(file == NULL){
        std::cout << "open file error !!! " << ss.str() << std::endl;
        std::abort();
    }
    fprintf(file,"%s\n", "%%MatrixMarket matrix coordinate pattern general");
    fprintf(file, "%d %d %d\n", row_, col_, off_diag_nnz_);
    for(label r = 0; r < row_; ++r){
        for(label index = off_diag_rowptr_[r]; index < off_diag_rowptr_[r+1]; ++index){
            label c = off_diag_colidx_[index];
            fprintf(file, "%d %d\n", r + 1, c + 1);
        }
    }
    fclose(file);
}


}


