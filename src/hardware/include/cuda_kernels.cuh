/**
 * HyVERX Hardware Acceleration CUDA Kernels Header
 * 
 * This header defines the interface for CUDA-optimized kernels for discrete GPU
 * acceleration, leveraging NVIDIA's CUDA platform for high-performance computing.
 * It provides specialized implementations of matrix operations, neural network
 * primitives, and Galois field arithmetic optimized for CUDA architectures.
 */

#ifndef HYVERX_CUDA_KERNELS_CUH
#define HYVERX_CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

// Constants for Tensor Core operations
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" {
    // Tensor Core matrix multiplication
    cudaError_t launch_tensor_core_matrix_multiply(
        const half* a_device,
        const half* b_device,
        float* c_device,
        int m,
        int n,
        int k,
        cudaStream_t stream = nullptr
    );
    
    // Standard matrix multiplication
    cudaError_t launch_matrix_multiply(
        const float* a_device,
        const float* b_device,
        float* c_device,
        int m,
        int n,
        int k,
        cudaStream_t stream = nullptr
    );
    
    // Neural network activation functions
    cudaError_t launch_relu_activation(
        const float* input_device,
        float* output_device,
        int size,
        cudaStream_t stream = nullptr
    );
    
    cudaError_t launch_sigmoid_activation(
        const float* input_device,
        float* output_device,
        int size,
        cudaStream_t stream = nullptr
    );
    
    cudaError_t launch_tanh_activation(
        const float* input_device,
        float* output_device,
        int size,
        cudaStream_t stream = nullptr
    );
    
    // Element-wise operations
    cudaError_t launch_add_constant(
        const float* input_device,
        float* output_device,
        int size,
        float constant,
        cudaStream_t stream = nullptr
    );
    
    cudaError_t launch_multiply_constant(
        const float* input_device,
        float* output_device,
        int size,
        float constant,
        cudaStream_t stream = nullptr
    );
    
    // Convolution operation
    cudaError_t launch_convolution(
        const float* input_device,
        const float* kernel_device,
        float* output_device,
        int batch_size,
        int input_height,
        int input_width,
        int input_channels,
        int kernel_height,
        int kernel_width,
        int output_channels,
        int output_height,
        int output_width,
        int stride_h,
        int stride_w,
        int padding_h,
        int padding_w,
        cudaStream_t stream = nullptr
    );
    
    // Galois field operations
    cudaError_t launch_gf_multiply(
        const unsigned short* a_device,
        const unsigned short* b_device,
        unsigned short* result_device,
        int size,
        const unsigned short* exp_table_device,
        const unsigned short* log_table_device,
        cudaStream_t stream = nullptr
    );
    
    cudaError_t launch_gf_add(
        const unsigned short* a_device,
        const unsigned short* b_device,
        unsigned short* result_device,
        int size,
        cudaStream_t stream = nullptr
    );
    
    // Syndrome calculation for Reed-Solomon codes
    cudaError_t launch_calculate_syndromes(
        const unsigned short* data_device,
        int data_length,
        unsigned short* syndromes_device,
        int syndrome_count,
        const unsigned short* exp_table_device,
        const unsigned short* log_table_device,
        cudaStream_t stream = nullptr
    );
    
    // Polynomial evaluation in batches
    cudaError_t launch_polynomial_eval_batch(
        const unsigned short* polys_device,
        const int* poly_lengths_device,
        int max_poly_len,
        int n_polys,
        const unsigned short* points_device,
        int n_points,
        unsigned short* results_device,
        const unsigned short* log_table_device,
        const unsigned short* exp_table_device,
        int field_size,
        cudaStream_t stream = nullptr
    );
}

// Utility function to check if CUDA is available
bool is_cuda_available();

// Utility function to get the number of CUDA devices
int get_cuda_device_count();

// Utility function to check if Tensor Cores are available
bool are_tensor_cores_available(int device_id = 0);

// Utility function to get CUDA device properties
cudaDeviceProp get_device_properties(int device_id = 0);

// Memory management utilities
template <typename T>
cudaError_t allocate_device_memory(T** device_ptr, size_t size);

template <typename T>
cudaError_t free_device_memory(T* device_ptr);

template <typename T>
cudaError_t copy_host_to_device(const T* host_ptr, T* device_ptr, size_t size);

template <typename T>
cudaError_t copy_device_to_host(const T* device_ptr, T* host_ptr, size_t size);

// CUDA stream management
cudaError_t create_stream(cudaStream_t* stream);
cudaError_t destroy_stream(cudaStream_t stream);
cudaError_t synchronize_stream(cudaStream_t stream);

#endif // HYVERX_CUDA_KERNELS_CUH