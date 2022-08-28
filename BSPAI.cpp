#include <CL/cl.h>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "BSPAI.hpp"
#include "OpenclKernels.hpp"

template <unsigned int block_size>
BSPAI<block_size>::BSPAI(const std::vector<int> &rowPointers_,
                         const std::vector<int> &colIndices_,
                         const std::vector<double> &nnzValues_,
                         const int fillIn_):
    rowPointers(rowPointers_),
    colIndices(colIndices_),
    nnzValues(nnzValues_),
    fillIn(fillIn_)
{
    bs = block_size;
    Nb = rowPointers.size() - 1;

    identityBlockIndex.resize(Nb);
    nbrows.resize(Nb);
    nbcols.resize(Nb);
    srpPointers.resize(Nb + 1);
    sciPointers.resize(Nb + 1);
    svlPointers.resize(Nb + 1);
    sqrPointers.resize(Nb + 1);
    rhsPointers.resize(Nb + 1);
    spaiColPointers.resize(Nb + 1);

    srpPointers[0] = 0;
    sciPointers[0] = 0;
    svlPointers[0] = 0;
    sqrPointers[0] = 0;
    rhsPointers[0] = 0;
    spaiColPointers[0] = 0;
}

template <unsigned int block_size>
void BSPAI<block_size>::setOpenCL(std::shared_ptr<cl::Context>& context_, std::shared_ptr<cl::CommandQueue>& queue_)
{
    context = context_;
    queue = queue_;
}

template <unsigned int block_size>
void BSPAI<block_size>::setIJSets(int col)
{
    jset.clear();

    auto fcol = colIndices.begin() + rowPointers[col];
    auto lcol = colIndices.begin() + rowPointers[col + 1];
    jset.insert(fcol, lcol);

    for(int f = 0; f <= fillIn; f++){
        iset.clear();

        for(auto it = jset.begin(); it != jset.end(); ++it){
            auto frow = colIndices.begin() + rowPointers[*it];
            auto lrow = colIndices.begin() + rowPointers[*it + 1];
            iset.insert(frow, lrow);
        }

        if(f < fillIn){
            jset = iset;
        }
    }

    nbrows[col] = iset.size();
    nbcols[col] = jset.size();

    spaiRowIndices.insert(spaiRowIndices.end(), jset.begin(), jset.end());
    spaiColPointers[col + 1] = spaiRowIndices.size();
}

template <unsigned int block_size>
void BSPAI<block_size>::setSMVecs(int col)
{
    smIndices.clear();
    smValsLocations.clear();
    smPointers.assign(iset.size() + 1, 0);

    unsigned int i = 1;
    for(auto rit = iset.begin(); rit != iset.end(); ++rit){
        auto fcol = colIndices.begin() + rowPointers[*rit];
        auto lcol = colIndices.begin() + rowPointers[*rit + 1];

        for(auto cit = fcol; cit != lcol; ++cit){
            if(jset.count(*cit)){
                smIndices.push_back(*cit);
                smValsLocations.resize(smValsLocations.size() + bs * bs);
                std::iota(smValsLocations.end() - bs * bs, smValsLocations.end(), (rowPointers[*rit] + cit - fcol) * bs * bs);
            }
        }

        smPointers[i] = smIndices.size();
        i++;
    }
}

template <unsigned int block_size>
void BSPAI<block_size>::gatherSMIndices()
{
    gather = smIndices;
    std::transform(gather.begin(), gather.end(), smIndices.begin(),
        [=](int i){return std::distance(jset.begin(), std::find(jset.begin(), jset.end(), i));});
}

template <unsigned int block_size>
void BSPAI<block_size>::buildSubmatrices()
{
    for(int col = 0; col < Nb; col++){
        setIJSets(col);
        setSMVecs(col);
        gatherSMIndices();

        identityBlockIndex[col] = std::distance(iset.begin(), std::find(iset.begin(), iset.end(), col));
        submatrixRowPointers.insert(submatrixRowPointers.end(), smPointers.begin(), smPointers.end());
        submatrixColIndices.insert(submatrixColIndices.end(), smIndices.begin(), smIndices.end());
        submatrixValsLocations.insert(submatrixValsLocations.end(), smValsLocations.begin(), smValsLocations.end());

        srpPointers[col + 1] = submatrixRowPointers.size();
        sciPointers[col + 1] = submatrixColIndices.size();
        svlPointers[col + 1] = submatrixValsLocations.size();
        sqrPointers[col + 1] = sqrPointers[col] + nbrows[col] * nbcols[col] * bs * bs;
        rhsPointers[col + 1] = rhsPointers[col] + nbrows[col] * bs * bs;
            
        spaiColPointers[col + 1] = spaiColPointers[col] + jset.size();
        spaiRowIndices.insert(spaiRowIndices.end(), jset.begin(), jset.end());
    }
}

template <unsigned int block_size>
void BSPAI<block_size>::writeDataToGPU()
{
    std::call_once(ocl_init, [&]() {
        d_nnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzValues.size());
        d_nbrows = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * nbrows.size());
        d_nbcols = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * nbcols.size());
        d_identityBlockIndex = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * identityBlockIndex.size());
        d_submatrixRowPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * submatrixRowPointers.size());
        d_submatrixColIndices = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * submatrixColIndices.size());
        d_submatrixValsLocations = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * submatrixValsLocations.size());
        d_submatrixQR = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * sqrPointers.back());
        d_rhs = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * rhsPointers.back());
        d_srpPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * srpPointers.size());
        d_sciPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * sciPointers.size());
        d_svlPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * svlPointers.size());
        d_sqrPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * sqrPointers.size());
        d_rhsPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * rhsPointers.size());
        d_spaiColPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * spaiColPointers.size());
        d_spaiRowIndices = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * spaiRowIndices.size());
        d_spaiNnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * spaiRowIndices.size() * bs * bs);

        events.resize(13);
        err = queue->enqueueWriteBuffer(d_nbrows, CL_FALSE, 0, nbrows.size() * sizeof(int), nbrows.data(), nullptr, &events[0]);
        err |= queue->enqueueWriteBuffer(d_nbcols, CL_FALSE, 0, nbcols.size() * sizeof(int), nbcols.data(), nullptr, &events[1]);
        err |= queue->enqueueWriteBuffer(d_identityBlockIndex, CL_FALSE, 0, identityBlockIndex.size() * sizeof(int), identityBlockIndex.data(), nullptr, &events[2]);
        err |= queue->enqueueWriteBuffer(d_submatrixRowPointers, CL_FALSE, 0, submatrixRowPointers.size() * sizeof(int), submatrixRowPointers.data(), nullptr, &events[3]);
        err |= queue->enqueueWriteBuffer(d_submatrixColIndices, CL_FALSE, 0, submatrixColIndices.size() * sizeof(int), submatrixColIndices.data(), nullptr, &events[4]);
        err |= queue->enqueueWriteBuffer(d_submatrixValsLocations, CL_FALSE, 0, submatrixValsLocations.size() * sizeof(int), submatrixValsLocations.data(), nullptr, &events[5]);
        err |= queue->enqueueWriteBuffer(d_srpPointers, CL_FALSE, 0, srpPointers.size() * sizeof(int), srpPointers.data(), nullptr, &events[6]);
        err |= queue->enqueueWriteBuffer(d_sciPointers, CL_FALSE, 0, sciPointers.size() * sizeof(int), sciPointers.data(), nullptr, &events[7]);
        err |= queue->enqueueWriteBuffer(d_svlPointers, CL_FALSE, 0, svlPointers.size() * sizeof(int), svlPointers.data(), nullptr, &events[8]);
        err |= queue->enqueueWriteBuffer(d_sqrPointers, CL_FALSE, 0, sqrPointers.size() * sizeof(int), sqrPointers.data(), nullptr, &events[9]);
        err |= queue->enqueueWriteBuffer(d_rhsPointers, CL_FALSE, 0, rhsPointers.size() * sizeof(int), rhsPointers.data(), nullptr, &events[10]);
        err |= queue->enqueueWriteBuffer(d_spaiColPointers, CL_FALSE, 0, spaiColPointers.size() * sizeof(int), spaiColPointers.data(), nullptr, &events[11]);
        err |= queue->enqueueWriteBuffer(d_spaiRowIndices, CL_FALSE, 0, spaiRowIndices.size() * sizeof(int), spaiRowIndices.data(), nullptr, &events[12]);
        cl::WaitForEvents(events);
    });

    events.resize(4);
    err = queue->enqueueWriteBuffer(d_nnzValues, CL_FALSE, 0, nnzValues.size() * sizeof(double), nnzValues.data(), nullptr, &events[0]);
    err |= queue->enqueueFillBuffer(d_submatrixQR, 0, 0, sqrPointers.back() * sizeof(double), nullptr, &events[1]);
    err |= queue->enqueueFillBuffer(d_rhs, 0, 0, rhsPointers.back() * sizeof(double), nullptr, &events[2]);
    err |= queue->enqueueFillBuffer(d_spaiNnzValues, 0, 0, spaiRowIndices.size() * bs * bs * sizeof(double), nullptr, &events[3]);
    cl::WaitForEvents(events);
}

template <unsigned int block_size>
void BSPAI<block_size>::QRDecomposititon()
{
    unsigned int bs = block_size;
    int max_nbcols = *std::max_element(nbcols.begin(), nbcols.end());

    for(int tile = 0; tile < max_nbcols; tile++){
        OpenclKernels::qr_decomposition(bs, Nb, tile, d_nbrows, d_nbcols, d_submatrixRowPointers, d_srpPointers,
                                        d_submatrixColIndices, d_sciPointers, d_submatrixValsLocations, d_svlPointers,
                                        d_sqrPointers, d_rhsPointers, d_identityBlockIndex, d_nnzValues, d_submatrixQR, d_rhs);
    }
}

template class BSPAI<3>;