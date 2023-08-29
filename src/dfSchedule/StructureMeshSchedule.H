#pragma once

#include "MeshSchedule.H"
#include <vector>
#include <tuple>

namespace Foam{

// 3D结构化网格分块图染色

class StructureMeshSchedule : public MeshSchedule{
private:
    label nCells_;

    std::vector<std::tuple<label,label,label>> points_;

    label x_dim_;
    label y_dim_;
    label z_dim_;

    labelList cell_scheduling_;
    labelList face_scheduling_;

    label blocksize_;
    labelList color_;

public:

    StructureMeshSchedule(int blocksize){
        blocksize_ = blocksize;
    };

    virtual void Setup(const fvMesh& mesh);

    virtual ~StructureMeshSchedule(){};

    const labelList& face_scheduling(){return face_scheduling;};
    
};


extern StructureMeshSchedule structureMeshSchedule;

}
