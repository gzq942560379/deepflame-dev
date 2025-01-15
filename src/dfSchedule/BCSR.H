#pragma once
#include <fvMesh.H>
#include <vector>
#include <utility>
#include "csrPattern.H"

#define bid2d(rbid, cbid) (rbid * rowBlockCount_ + cbid)

namespace Foam{

class BCSR{
private:
    
    label n_;

    label rowBlockCount_;
    labelList rowBlockPtr_;

    


public:

    BCSR(const csrPattern csr, const labelList& regionPtr);

    ~BCSR(){};


};

}
