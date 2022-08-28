#ifndef OPENCL_KERNELS_HPP
#define OPENCL_KERNELS_HPP

#include <string>
#include <memory>
#include "opencl.hpp"

using qr_decomposition_kernel_type = cl::KernelFunctor<const int, const int, const int, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                       cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&,
                                                       cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg>;

class OpenclKernels
{
private:
    static int verbosity;
    static cl::CommandQueue *queue;
    static bool initialized;

    static std::unique_ptr<qr_decomposition_kernel_type> qr_decomposition_k;

    OpenclKernels(){}; // diasable instantiation

public:
    static const std::string qr_decomposition_str;

    static void init(cl::Context *context, cl::CommandQueue *queue, std::vector<cl::Device>& devices);
    static void qr_decomposition(int block_size, int Nb, int tile, cl::Buffer nbrows, cl::Buffer nbcols, cl::Buffer srp,
                                 cl::Buffer srpp, cl::Buffer sci, cl::Buffer scip, cl::Buffer svl, cl::Buffer svlp, cl::Buffer sqrp,
                                 cl::Buffer rhsp, cl::Buffer ibid, cl::Buffer vals, cl::Buffer QR, cl::Buffer qx);
};

#endif
