/**
 * HyVERX Hardware Acceleration AVX2 Kernels Header
 * 
 * This header defines the interface for AVX2-optimized kernels for CPU-based 
 * hardware acceleration operations.
 */

#ifndef HYVERX_AVX2_KERNELS_HPP
#define HYVERX_AVX2_KERNELS_HPP

#include <cstdint>
#include <immintrin.h>

namespace hardware {
namespace avx2 {

// Enumeration of element-wise operations
enum ElementWiseOperation {
    RELU,
    SIGMOID,
    TANH,
    ADD,
    MULTIPLY
};

// Check if AVX2 is supported
bool is_avx2_supported();

// Matrix multiplication
void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k);

// Element-wise operations
void element_wise_operation(const float* input, float* output, int size, ElementWiseOperation op, float constant = 0.0f);

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
    int padding_w
);

// Galois field multiplication
void gf_multiply(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size
);

// Galois field addition
void gf_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size
);

// Calculate syndromes for Reed-Solomon codes
void calculate_syndromes(
    const uint16_t* data,
    int data_length,
    uint16_t* syndromes,
    int syndrome_count,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size
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
    int field_size
);

// Helper functions (implementation details)
__m256 exp_approx_avx2(__m256 x);
__m256 sigmoid_approx_avx2(__m256 x);

} // namespace avx2
} // namespace hardware

#endif // HYVERX_AVX2_KERNELS_HPP