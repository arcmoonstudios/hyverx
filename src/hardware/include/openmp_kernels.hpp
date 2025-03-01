/**
 * HyVERX Hardware Acceleration OpenMP Kernels Header
 * 
 * This header defines the interface for OpenMP-parallelized kernels for CPU-based 
 * hardware acceleration operations.
 */

#ifndef HYVERX_OPENMP_KERNELS_HPP
#define HYVERX_OPENMP_KERNELS_HPP

#include <cstdint>

namespace hardware {
namespace openmp {

// Enumeration of element-wise operations
enum ElementWiseOperation {
    RELU,
    SIGMOID,
    TANH,
    ADD,
    MULTIPLY
};

// Matrix multiplication
void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k, int num_threads = 0);

// Element-wise operations
void element_wise_operation(const float* input, float* output, int size, ElementWiseOperation op, float constant = 0.0f, int num_threads = 0);

// Convolution
void convolution(
    const float* input,
    const float* kernel,
    float* output,
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
    int num_threads = 0
);

// Galois field multiplication
void gf_multiply(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size,
    int num_threads = 0
);

// Galois field addition
void gf_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    int num_threads = 0
);

// Calculate syndromes for Reed-Solomon codes
void calculate_syndromes(
    const uint16_t* data,
    int data_length,
    uint16_t* syndromes,
    int syndrome_count,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size,
    int num_threads = 0
);

// Polynomial evaluation in batches
void polynomial_eval_batch(
    const uint16_t* polys,
    const int* poly_lengths,
    int max_poly_len,
    int n_polys,
    const uint16_t* points,
    int n_points,
    uint16_t* results,
    const uint16_t* log_table,
    const uint16_t* exp_table,
    int field_size,
    int num_threads = 0
);

// Helper functions
int get_optimal_thread_count();

} // namespace openmp
} // namespace hardware

#endif // HYVERX_OPENMP_KERNELS_HPP