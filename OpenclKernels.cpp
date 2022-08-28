#include <chrono>
#include <iostream>
#include "OpenclKernels.hpp"

// define static variables and kernels
int OpenclKernels::verbosity;
cl::CommandQueue *OpenclKernels::queue;
bool OpenclKernels::initialized = false;

std::unique_ptr<qr_decomposition_kernel_type> OpenclKernels::qr_decomposition_k;

// divide A by B, and round up: return (int)ceil(A/B)
unsigned int ceilDivision(const unsigned int A, const unsigned int B)
{
    return A / B + (A % B > 0);
}

void OpenclKernels::init(cl::Context *context, cl::CommandQueue *queue_, std::vector<cl::Device>& devices)
{
if (initialized) {
        std::cout << "Warning OpenclKernels is already initialized" << std::endl;
        return;
    }

    queue = queue_;

    cl::Program::Sources sources;
    sources.emplace_back(qr_decomposition_str);

    cl::Program program = cl::Program(*context, sources);
    program.build(devices);

    qr_decomposition_k.reset(new qr_decomposition_kernel_type(cl::Kernel(program, "qr_decomposition")));

    initialized = true;
    verbosity = 0;
}

void OpenclKernels::qr_decomposition(int block_size, int Nb, int tile, cl::Buffer nbrows, cl::Buffer nbcols, cl::Buffer srp,
                                     cl::Buffer srpp, cl::Buffer sci, cl::Buffer scip, cl::Buffer svl, cl::Buffer svlp, cl::Buffer sqrp,
                                     cl::Buffer rhsp, cl::Buffer ibid, cl::Buffer vals, cl::Buffer QR, cl::Buffer qx)
{
    const unsigned int work_group_size = 32;
    const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
    const unsigned int total_work_items = num_work_groups * work_group_size;
    const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

    cl::Event event;

    auto start = std::chrono::high_resolution_clock::now();
    event = (*qr_decomposition_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                  block_size, Nb, tile, nbrows, nbcols, srp, srpp, sci, scip, svl, svlp, sqrp, rhsp, ibid,
                                  vals, QR, qx, cl::Local(lmem_per_work_group));
    auto end = std::chrono::high_resolution_clock::now();

    if(verbosity > 0){
        double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        time_taken *= 1e-6;
        std::cout << "OpenclKernels::qr_decomposition took " << std::fixed << time_taken << " ms" << std::endl;
    }
}