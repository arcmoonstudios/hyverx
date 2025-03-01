#include <iostream>
#include <vector>
#include <CL/cl.h>

// Simple error checking
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL error: " << err << " at line " << __LINE__ << std::endl; \
        return 1; \
    }

// OpenCL kernel as a string
const char* kernelSource = R"(
__kernel void vectorAdd(__global const float* a, 
                         __global const float* b,
                         __global float* c) {
    int index = get_global_id(0);
    c[index] = a[index] + b[index];
}
)";

int main() {
    std::cout << "OpenCL Test Program" << std::endl;
    
    // Data size
    const int DATA_SIZE = 10;
    
    // Host data
    std::vector<float> a(DATA_SIZE, 1.0f);
    std::vector<float> b(DATA_SIZE, 2.0f);
    std::vector<float> c(DATA_SIZE, 0.0f);
    
    // OpenCL variables
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferC;
    cl_int err;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    
    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    CHECK_ERROR(err);
    
    // Print device info
    char deviceName[128];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    CHECK_ERROR(err);
    std::cout << "Using device: " << deviceName << std::endl;
    
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    
    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
    
    // Create buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                             sizeof(float) * DATA_SIZE, a.data(), &err);
    CHECK_ERROR(err);
    
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                             sizeof(float) * DATA_SIZE, b.data(), &err);
    CHECK_ERROR(err);
    
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                             sizeof(float) * DATA_SIZE, NULL, &err);
    CHECK_ERROR(err);
    
    // Create program
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_ERROR(err);
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log if there's an error
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        std::cerr << "Build error: " << buildLog.data() << std::endl;
        return 1;
    }
    
    // Create kernel
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERROR(err);
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    CHECK_ERROR(err);
    
    // Execute kernel
    size_t globalSize = DATA_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Read results
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * DATA_SIZE, c.data(), 0, NULL, NULL);
    CHECK_ERROR(err);
    
    // Wait for completion
    clFinish(queue);
    
    // Print results
    std::cout << "Vector addition results: ";
    for (int i = 0; i < DATA_SIZE; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
} 