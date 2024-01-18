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

public:

    StructureMeshSchedule(label dummy){};

    virtual void Setup(const fvMesh& mesh);

    virtual ~StructureMeshSchedule(){};

    const labelList& face_scheduling(){return face_scheduling_;};

    label index(label x, label y, label z);
    void xyz(label index, label& x, label& y, label& z);
    
};


extern StructureMeshSchedule structureMeshSchedule;

}
