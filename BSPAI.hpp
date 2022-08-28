#ifndef SPAI_HPP
#define SPAI_HPP

#include <set>
#include <vector>
#include <mutex>
#include <memory>
#include "opencl.hpp"

template <unsigned int block_size>
class BSPAI
{
private:
    int fillIn, Nb, bs;

    std::set<int> iset, jset;

    std::vector<int> nbrows, nbcols;
    std::vector<int> rowPointers, colIndices;
    std::vector<double> nnzValues;
    std::vector<int> smPointers, smIndices, smValsLocations, gather;
    std::vector<int> identityBlockIndex;
    std::vector<int> submatrixRowPointers, submatrixColIndices, submatrixValsLocations;
    std::vector<int> srpPointers; // pointers for submatrixRowPointers
    std::vector<int> sciPointers; // pointers for submatrixColIndices
    std::vector<int> svlPointers; // pointers for submatrixValsLocations
    std::vector<int> sqrPointers; // pointers for submatrixQR
    std::vector<int> rhsPointers; // pointers for LSQ systems' right-hand-sides 
    std::vector<int> spaiColPointers, spaiRowIndices;
    std::vector<double> spaiNnzValues;

    std::once_flag ocl_init;

    cl_int err;
    std::vector<cl::Event> events;
    std::shared_ptr<cl::Context> context;
    std::shared_ptr<cl::CommandQueue> queue;
    cl::Buffer d_nnzValues;
    cl::Buffer d_nbrows;
    cl::Buffer d_nbcols;
    cl::Buffer d_identityBlockIndex;
    cl::Buffer d_submatrixRowPointers;
    cl::Buffer d_submatrixColIndices;
    cl::Buffer d_submatrixValsLocations;
    cl::Buffer d_submatrixQR;
    cl::Buffer d_rhs;
    cl::Buffer d_srpPointers;
    cl::Buffer d_sciPointers;
    cl::Buffer d_svlPointers;
    cl::Buffer d_sqrPointers;
    cl::Buffer d_rhsPointers;
    cl::Buffer d_spaiColPointers;
    cl::Buffer d_spaiRowIndices;
    cl::Buffer d_spaiNnzValues;

    void setIJSets(int col);
    void setSMVecs(int col);
    void gatherSMIndices();

public:
    BSPAI(const std::vector<int> &rowPointers_,
          const std::vector<int> &colIndices_,
          const std::vector<double> &nnzValues_,
          const int fillIn_);
    
    void setOpenCL(std::shared_ptr<cl::Context>& context_, std::shared_ptr<cl::CommandQueue>& queue_);
    void buildSubmatrices();
    void writeDataToGPU();
    void QRDecomposititon();
};

#endif
