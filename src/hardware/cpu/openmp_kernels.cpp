/**
 * HyVERX Hardware Acceleration OpenMP Kernels
 * 
 * This module provides OpenMP implementations of kernels for CPU-based 
 * hardware acceleration, focusing on parallel execution of neural network
 * operations and Galois field arithmetic for error correction codes.
 * 
 * Key features:
 * - Multi-threaded matrix operations
 * - Parallel neural network primitives
 * - Concurrent Galois field operations
 * - Efficient thread management and workload distribution
 */

#include "../include/openmp_kernels.hpp"

#include <omp.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstring>

namespace hardware {
namespace openmp {

void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k, int num_threads) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Perform parallel matrix multiplication
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void element_wise_operation(const float* input, float* output, int size, ElementWiseOperation op, float constant, int num_threads) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Perform parallel element-wise operation
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        switch (op) {
            case RELU:
                output[i] = std::max(0.0f, input[i]);
                break;
            case SIGMOID:
                output[i] = 1.0f / (1.0f + std::exp(-input[i]));
                break;
            case TANH:
                output[i] = std::tanh(input[i]);
                break;
            case ADD:
                output[i] = input[i] + constant;
                break;
            case MULTIPLY:
                output[i] = input[i] * constant;
                break;
        }
    }
}

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
    int num_threads
) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Parallelize over batches and output channels
    #pragma omp parallel for collapse(2)
    for (int batch = 0; batch < batch_size; batch++) {
        for (int out_c = 0; out_c < output_channels; out_c++) {
            // Process each output pixel
            for (int out_y = 0; out_y < output_height; out_y++) {
                for (int out_x = 0; out_x < output_width; out_x++) {
                    float sum = 0.0f;
                    
                    // Convolve kernel with input at this position
                    for (int ky = 0; ky < kernel_height; ky++) {
                        int in_y = out_y * stride_h - padding_h + ky;
                        
                        if (in_y >= 0 && in_y < input_height) {
                            for (int kx = 0; kx < kernel_width; kx++) {
                                int in_x = out_x * stride_w - padding_w + kx;
                                
                                if (in_x >= 0 && in_x < input_width) {
                                    for (int ic = 0; ic < input_channels; ic++) {
                                        float in_val = input[((batch * input_height + in_y) * input_width + in_x) * input_channels + ic];
                                        float kernel_val = kernel[((ky * kernel_width + kx) * input_channels + ic) * output_channels + out_c];
                                        sum += in_val * kernel_val;
                                    }
                                }
                            }
                        }
                    }
                    
                    output[((batch * output_height + out_y) * output_width + out_x) * output_channels + out_c] = sum;
                }
            }
        }
    }
}

void gf_multiply(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size,
    int num_threads
) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Perform parallel Galois field multiplication
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        uint16_t a_val = a[i];
        uint16_t b_val = b[i];
        
        if (a_val == 0 || b_val == 0) {
            result[i] = 0;
        } else {
            uint16_t log_a = log_table[a_val];
            uint16_t log_b = log_table[b_val];
            uint16_t log_sum = log_a + log_b;
            if (log_sum >= field_size - 1) {
                log_sum -= field_size - 1;
            }
            result[i] = exp_table[log_sum];
        }
    }
}

void gf_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    int num_threads
) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Perform parallel Galois field addition
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result[i] = a[i] ^ b[i];
    }
}

void calculate_syndromes(
    const uint16_t* data,
    int data_length,
    uint16_t* syndromes,
    int syndrome_count,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size,
    int num_threads
) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Parallelize over syndromes
    #pragma omp parallel for
    for (int i = 0; i < syndrome_count; i++) {
        uint16_t sum = 0;
        uint16_t alpha_i = exp_table[i + 1];
        uint16_t alpha_power = 1;
        
        // Horner's method for polynomial evaluation
        for (int j = 0; j < data_length; j++) {
            // data[j] * alpha_power
            uint16_t term = 0;
            if (data[j] != 0 && alpha_power != 0) {
                uint16_t log_data = log_table[data[j]];
                uint16_t log_alpha = log_table[alpha_power];
                uint16_t log_sum = log_data + log_alpha;
                if (log_sum >= field_size - 1) {
                    log_sum -= field_size - 1;
                }
                term = exp_table[log_sum];
            }
            
            sum ^= term;
            
            // Update alpha_power = alpha_power * alpha_i
            if (alpha_power != 0) {
                uint16_t log_alpha_power = log_table[alpha_power];
                uint16_t log_alpha_i = log_table[alpha_i];
                uint16_t log_sum = log_alpha_power + log_alpha_i;
                if (log_sum >= field_size - 1) {
                    log_sum -= field_size - 1;
                }
                alpha_power = exp_table[log_sum];
            }
        }
        
        syndromes[i] = sum;
    }
}

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
    int num_threads
) {
    // Set the number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Parallelize over polynomial and point combinations
    #pragma omp parallel for collapse(2)
    for (int poly_idx = 0; poly_idx < n_polys; poly_idx++) {
        for (int point_idx = 0; point_idx < n_points; point_idx++) {
            uint16_t point = points[point_idx];
            int poly_len = poly_lengths[poly_idx];
            int poly_offset = poly_idx * max_poly_len;
            
            // Use Horner's method for polynomial evaluation
            uint16_t result = 0;
            
            if (poly_len > 0) {
                // Start with the highest degree coefficient
                result = polys[poly_offset + poly_len - 1];
                
                // Apply Horner's method
                for (int i = poly_len - 2; i >= 0; i--) {
                    if (result == 0) {
                        // If result is 0, no need to multiply
                        result = polys[poly_offset + i];
                    } else {
                        // result = result * point + polys[poly_offset + i]
                        if (point != 0) {
                            // Multiply result by point using log tables
                            uint16_t log_result = log_table[result];
                            uint16_t log_point = log_table[point];
                            uint16_t log_sum = log_result + log_point;
                            if (log_sum >= field_size - 1) {
                                log_sum -= field_size - 1;
                            }
                            result = exp_table[log_sum];
                        } else {
                            result = 0;
                        }
                        
                        // Add the current coefficient (XOR in GF(2^m))
                        result ^= polys[poly_offset + i];
                    }
                }
            }
            
            results[poly_idx * n_points + point_idx] = result;
        }
    }
}

// Helper functions for OpenMP parallelization

int get_optimal_thread_count() {
    // Get the number of available hardware threads
    int num_threads = omp_get_max_threads();
    
    // Limit to a reasonable number if the hardware reports an extremely high value
    if (num_threads > 64) {
        num_threads = 64;
    }
    
    return num_threads;
}

} // namespace openmp
} // namespace hardware