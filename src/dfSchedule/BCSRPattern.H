#pragma once
#include <fvMesh.H>
#include <vector>
#include <utility>
#include "csrPattern.H"

#define bid2d(rbid, cbid) (rbid * rowBlockCount_ + cbid)

namespace Foam{

class BCSRPattern{
private:
    
    label n_;

    label rowBlockCount_;
    labelList rowBlockPtr_;


    std::vector<std::vector<std::pair<label, label>>> blocks_;
    


public:

    BCSRPattern(const csrPattern csr, const labelList& regionPtr);

    ~BCSRPattern(){};

    void show() const;

};

}
