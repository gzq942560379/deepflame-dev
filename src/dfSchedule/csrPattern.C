#include <vector>
#include <cassert>
#include <string>
#include <PstreamGlobals.H>
#include "csrPattern.H"

namespace Foam{

csrPattern::csrPattern(const fvMesh& mesh){
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    assert(owner.size() == neighbour.size());
    this->n_ = mesh.nCells();
    label face_count = neighbour.size();
    this->nnz_ = face_count * 2 + n_;

    this->rowptr_.resize(n_ + 1);
    this->colidx_.resize(nnz_);

    // compute row_count
    std::vector<label> row_count(n_, 1);
    std::vector<label> current_index(n_ + 1);
    for(label i = 0; i < face_count; ++i){
        auto own = owner[i];
        auto nei = neighbour[i];
        row_count[own] += 1;
        row_count[nei] += 1;
    }

    rowptr_[0] = 0;
    current_index[0] = 0;
    for(label i = 0; i < n_; ++i){
        rowptr_[i + 1] = rowptr_[i] + row_count[i];
        current_index[i + 1] = rowptr_[i] + row_count[i];
    }

    // lower
    // (nei, own)
    for(label i = 0; i < face_count; ++i){
        label r = neighbour[i];
        label c = owner[i];
        label index = current_index[r];
        // face2lower_[i] = index;
        colidx_[index] = c;
        current_index[r] += 1;
    }

    // diag
    for(label rc = 0; rc < n_; ++rc){
        label index = current_index[rc];
        colidx_[index] = rc;
        current_index[rc] += 1;
    }

    // upper
    // (own, nei)
    for(label i = 0; i < face_count; ++i){
        label r = owner[i];
        label c = neighbour[i];
        label index = current_index[r];
        // face2upper_[i] = index;
        colidx_[index] = c;
        current_index[r] += 1;
    }
}

csrPattern csrPattern::blocking(label block_size){
    csrPattern blocked_matrix;
    assert(n_ % block_size == 0);

    label block_count = n_ / block_size;

    blocked_matrix.n_ = block_count;

    std::vector<label> mark(block_count, -1);
    std::vector<label> row_count(block_count, 0);

    // row count
    for(label br = 0; br < block_count; ++br){
        label bs = br * block_size;
        label be = (br + 1) * block_size;
        for(label r = bs; r < be; ++r){
            for(label index = rowptr_[r]; index < rowptr_[r+1]; ++index){
                label c = colidx_[index];
                label bc = c / block_size;
                mark[bc] = br;
            }
        }
        for(label bc = 0; bc < block_count; ++bc){
            if(mark[bc] == br){
                row_count[br] += 1;
            }
        }
    }

    std::vector<label> cur_index(block_count);
    blocked_matrix.rowptr_.resize(block_count + 1);

    // rowptr nnz
    blocked_matrix.rowptr_[0] = 0;
    for(label br = 0; br < block_count; ++br){
        blocked_matrix.rowptr_[br + 1] = blocked_matrix.rowptr_[br] + row_count[br];
    }
    blocked_matrix.nnz_ = blocked_matrix.rowptr_[block_count];
    blocked_matrix.colidx_.resize(blocked_matrix.nnz_);

    for(label br = 0; br < block_count; ++br){
        cur_index[br] = blocked_matrix.rowptr_[br];
    }

    // reset mark
    for(label br = 0; br < block_count; ++br){
        mark[br] = -1;
    }

    // fill colidx
    for(label br = 0; br < block_count; ++br){
        label bs = br * block_size;
        label be = (br + 1) * block_size;
        for(label r = bs; r < be; ++r){
            for(label index = rowptr_[r]; index < rowptr_[r+1]; ++index){
                label c = colidx_[index];
                label bc = c / block_size;
                if(mark[bc] != br){
                    label bindex = cur_index[br];
                    blocked_matrix.colidx_[bindex] = bc;
                    cur_index[br] += 1;
                    mark[bc] = br;
                }
            }
        }
    }

    // check
    for(label br = 0; br < block_count; ++br){
        assert(cur_index[br] == blocked_matrix.rowptr_[br + 1]);
    }

    return blocked_matrix;
}

void csrPattern::write_mtx(const std::string& filename) const {
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
        fprintf(fw, "%ld %ld %ld\n", n_, n_, nnz_);
        for(label r = 0; r < n_; ++r){
            for(label index = rowptr_[r]; index < rowptr_[r+1]; ++index){
                label c = colidx_[index];
                fprintf(fw, "%ld %ld\n", r + 1, c + 1);
            }
        }
        fclose(fw);
    }
}

void csrPattern::order_natrue(std::vector<label>& order){
    order.resize(n_);
    for(label i = 0; i < n_; ++i){
        order[i] = i;
    }
}

void csrPattern::order_longest_row(std::vector<label>& order){
    std::vector<std::tuple<label,label>> row_length_tuples(n_);
    for(label r = 0; r < n_; ++r){
        row_length_tuples[r] = std::make_tuple(r, rowptr_[r+1] - rowptr_[r]);
    }
    // 按 行长 从大到小 行号 从小到大 排序
    std::sort(row_length_tuples.begin(), row_length_tuples.end(), [](const auto& a, const auto& b){
        return std::get<1>(a) == std::get<1>(b) ? std::get<0>(a) < std::get<0>(b) : std::get<1>(a) > std::get<1>(b);
    });
    order.resize(n_);
    for(label r = 0; r < n_; ++r){
        order[r] = std::get<0>(row_length_tuples[r]);
    }
}

void csrPattern::coloring(){
    Info << "coloring start" << endl;
    max_color_ = 0;

    color_.resize(n_);
    for(label i = 0; i < n_; ++i){
        color_[i] = -1;
    }

    std::vector<label> travel_order;
    // order_natrue(travel_order);
    order_longest_row(travel_order);


    std::vector<label> mark(n_, -1);

    for(label i = 0; i < n_; ++i){
        label u = travel_order[i];
        for(label index = rowptr_[u]; index < rowptr_[u+1]; ++index){
            label v = colidx_[index];
            label c = color_[v];
            if(c != -1){
                // i的邻居已经用过颜色c了
                mark[c] = i;
            }
        }
        label c = 0;
        while(c < max_color_ && mark[c] == i){
            c += 1;
        }
        if(c == max_color_){
            max_color_ += 1;
        }
        color_[u] = c;
    }

    for(label i = 0; i < n_; ++i){
        assert(color_[i] >= 0);
        assert(color_[i] < max_color_);
    }

    Info << "coloring end" << endl;
    Info << "max color : " << max_color_ << endl;

    color_count_.resize(max_color_);
    for(label c = 0; c < max_color_; ++c){
        color_count_[c] = 0;
    }
    for(label i = 0; i < n_; ++i){
        color_count_[color_[i]] += 1;
    }

    Info << "color count : " << endl;
    for(label c = 0; c < max_color_; ++c){
        Info << c << " : " << color_count_[c] << endl;
    }
}



}
