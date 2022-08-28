#include <vector>
#include <fstream>
#include <iostream>
#include <memory>

#include "OpenclKernels.hpp"
#include "BSPAI.hpp"

template<typename T>
void read_vec(const std::string &fname, std::vector<T> &temp){
    T value;
    std::ifstream input(fname.c_str());

    while(input >> value){
        temp.push_back(value);
    }
    input.close();
}

void initOpenCL(std::shared_ptr<cl::Context> &context, std::shared_ptr<cl::CommandQueue> &queue){
    int platformID = 0;
    int deviceID = 0;
    cl_int err = CL_SUCCESS;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Platform::get(&platforms);
    platforms[platformID].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    context = std::make_shared<cl::Context>(devices[deviceID]);
    queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));

    OpenclKernels::init(context.get(), queue.get(), devices);
}

int main(int argc, char** argv){
    std::vector<int> rowPointers, colIndices;
    std::vector<double> nnzValues;

    std::cout << "Starting program..." << std::endl;

    std::shared_ptr<cl::Context> context;
    std::shared_ptr<cl::CommandQueue> queue;
    initOpenCL(context, queue);
    
    read_vec<int>("/home/ebueno/OldDrive/mysrc/spai_v2/data/spe1case1/colIndices.txt", colIndices);
    read_vec<int>("/home/ebueno/OldDrive/mysrc/spai_v2/data/spe1case1/rowPointers.txt", rowPointers);
    read_vec<double>("/home/ebueno/OldDrive/mysrc/spai_v2/data/spe1case1/nnzValues.txt", nnzValues);

    BSPAI<3> spai(rowPointers, colIndices, nnzValues, 1);
    spai.setOpenCL(context, queue);
    spai.buildSubmatrices();
    spai.writeDataToGPU();

    return 0;
}
