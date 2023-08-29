#pragma once

#include <fvMesh.H>

namespace Foam{


class MeshSchedule{
private:

public:

    MeshSchedule(){};

    virtual ~MeshSchedule(){};

    virtual void Setup(const fvMesh& mesh){};

};

}
