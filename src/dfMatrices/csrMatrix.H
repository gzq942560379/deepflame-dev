#pragma once
#include <fvMesh.H>
#include <vector>
#include <string>

namespace Foam{

class csrMatrix{
private:
    const fvMesh& mesh_;
    //- Coefficients (not including interfaces)
    label row_;
    label col_;
    label off_diag_nnz_;
    std::vector<scalar> diag_value_;
    std::vector<label> off_diag_rowptr_;
    std::vector<label> off_diag_colidx_;
    std::vector<scalar> off_diag_value_;

    // map from face to lower in off_diag_value_
    // lower (nei,own)
    std::vector<label> face2lower_;
    // map from face to upper in off_diag_value_
    // lower (own,nei)
    std::vector<label> face2upper_;
    
public:
    // Static data
    // Declare name of the class and its debug switch
    ClassName("csrMatrix");

    // Constructors
    csrMatrix(const fvMesh& mesh);

    // Copy constructor
    csrMatrix(const csrMatrix& other);

    //- Destructor
    ~csrMatrix();


    void write_pattern(const std::string& filename);
};

}

