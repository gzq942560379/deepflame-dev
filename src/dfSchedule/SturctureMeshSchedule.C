#include "StructureMeshSchedule.H"

void Foam::StructureMeshSchedule::Setup(const fvMesh& mesh){
    nCells_ = mesh.nCells();
    
    const labelUList& face_ptr = mesh.lduAddr().ownerStartAddr();
    const labelUList& l = mesh.lduAddr().lowerAddr();
    const labelUList& u = mesh.lduAddr().upperAddr();

    const auto& cellCentres = mesh.cellCentres();
    forAll(cellCentres, i){
        points_.push_back(std::make_tuple(static_cast<label>(cellCentres[i][0] * 1e6), static_cast<label>(cellCentres[i][1] * 1e6), static_cast<label>(cellCentres[i][2] * 1e6)));
    }

    x_dim_ = 1;
    for(size_t i = 1; i < points_.size(); ++i){
        if(std::get<1>(points_[i]) == std::get<1>(points_[i-1])){
            x_dim_ += 1;
        }else{
            break;
        }
    }

    y_dim_ = 1;
    for(size_t i = 1; i < points_.size(); ++i){
        if(std::get<2>(points_[i]) == std::get<2>(points_[i-1])){
            y_dim_ += 1;
        }else{
            break;
        }
    }
    
    y_dim_ /= x_dim_;
    z_dim_ = nCells_ / y_dim_ / x_dim_;

    cell_scheduling_.resize(z_dim_ + 1);
    face_scheduling_.resize(z_dim_ + 1);

    cell_scheduling_[0] = 0;
    face_scheduling_[0] = 0;

    for(label i = 0 ; i < z_dim_; i++){
        cell_scheduling_[i+1] = cell_scheduling_[i] + (x_dim_ * y_dim_);
        face_scheduling_[i+1] = face_ptr[cell_scheduling_[i+1]];
    }
}



Foam::StructureMeshSchedule Foam::structureMeshSchedule(1);

