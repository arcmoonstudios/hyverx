/**
 * HyVERX Hardware Acceleration AVX2 Kernels
 * 
 * This module provides AVX2 (Advanced Vector Extensions 2) implementations
 * of kernels for CPU-based hardware acceleration, focusing on neural network
 * operations and Galois field arithmetic for error correction codes.
 * 
 * Key features:
 * - AVX2 optimized matrix operations
 * - Neural network primitives
 * - Galois field operations
 * - Efficient SIMD-based implementations
 */

#include "../include/avx2_kernels.hpp"

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace hardware {
namespace avx2 {

bool is_avx2_supported() {
    #ifdef __AVX2__
        // Check for AVX2 support at runtime
        int cpu_info[4];
        
        // Use CPUID to get CPU features
        // EAX=7, ECX=0 for extended features
        __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
        
        // AVX2 is indicated by bit 5 of EBX
        return (cpu_info[1] & (1 << 5)) != 0;
    #else
        return false;
    #endif
}

void matrix_multiply_avx2(const float* a, const float* b, float* c, int m, int n, int k) {
    // Ensure proper alignment for optimal AVX2 performance
    assert(reinterpret_cast<uintptr_t>(a) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(b) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(c) % 32 == 0);
    
    // Perform matrix multiplication with AVX2 optimization
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 8) {
            // Handle boundary case
            if (j + 8 > n) {
                // Fallback to scalar multiplication for edge cases
                for (int jj = j; jj < n; jj++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += a[i * k + p] * b[p * n + jj];
                    }
                    c[i * n + jj] = sum;
                }
                break;
            }
            
            // Initialize accumulator to zero
            __m256 sum = _mm256_setzero_ps();
            
            // Multiply and accumulate
            for (int p = 0; p < k; p++) {
                // Broadcast a[i, p] to all elements
                __m256 a_val = _mm256_set1_ps(a[i * k + p]);
                
                // Load 8 elements from b[p, j:j+8]
                __m256 b_val = _mm256_loadu_ps(&b[p * n + j]);
                
                // sum += a[i, p] * b[p, j:j+8]
                sum = _mm256_fmadd_ps(a_val, b_val, sum);
            }
            
            // Store the result
            _mm256_storeu_ps(&c[i * n + j], sum);
        }
    }
}

void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k) {
    // Check if AVX2 is supported
    if (is_avx2_supported() && m >= 8 && n >= 8 && k >= 8) {
        matrix_multiply_avx2(a, b, c, m, n, k);
    } else {
        // Fallback to scalar implementation
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
}

void relu_avx2(const float* input, float* output, int size) {
    // Process 8 elements at a time with AVX2
    int i = 0;
    
    // Vector with all zeros for comparison
    __m256 zeros = _mm256_setzero_ps();
    
    for (; i <= size - 8; i += 8) {
        // Load 8 elements from input
        __m256 in = _mm256_loadu_ps(&input[i]);
        
        // ReLU: max(0, x)
        __m256 result = _mm256_max_ps(zeros, in);
        
        // Store result
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void sigmoid_avx2(const float* input, float* output, int size) {
    // Process 8 elements at a time with AVX2
    int i = 0;
    
    // Constants for sigmoid approximation
    __m256 ones = _mm256_set1_ps(1.0f);
    __m256 neg_ones = _mm256_set1_ps(-1.0f);
    
    for (; i <= size - 8; i += 8) {
        // Load 8 elements from input
        __m256 in = _mm256_loadu_ps(&input[i]);
        
        // Sigmoid: 1 / (1 + exp(-x))
        // Multiply by -1
        __m256 neg_in = _mm256_mul_ps(in, neg_ones);
        
        // Compute exp(-x) using intrinsic approximation
        // First, restrict to a reasonable range to avoid overflow/underflow
        __m256 clamped = _mm256_max_ps(_mm256_set1_ps(-88.0f), _mm256_min_ps(neg_in, _mm256_set1_ps(88.0f)));
        
        // Approximate exp with several operations
        __m256 exp_result = exp_approx_avx2(clamped);
        
        // Calculate 1 + exp(-x)
        __m256 denom = _mm256_add_ps(ones, exp_result);
        
        // Calculate 1 / (1 + exp(-x))
        __m256 result = _mm256_div_ps(ones, denom);
        
        // Store result
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void tanh_avx2(const float* input, float* output, int size) {
    // Process 8 elements at a time with AVX2
    int i = 0;
    
    // Constants for tanh approximation
    __m256 twos = _mm256_set1_ps(2.0f);
    __m256 ones = _mm256_set1_ps(1.0f);
    
    for (; i <= size - 8; i += 8) {
        // Load 8 elements from input
        __m256 in = _mm256_loadu_ps(&input[i]);
        
        // tanh(x) = 2*sigmoid(2*x) - 1
        // Calculate 2*x
        __m256 two_x = _mm256_mul_ps(in, twos);
        
        // Calculate sigmoid(2*x)
        __m256 sig_result = sigmoid_approx_avx2(two_x);
        
        // Calculate 2*sigmoid(2*x)
        __m256 two_sig = _mm256_mul_ps(sig_result, twos);
        
        // Calculate 2*sigmoid(2*x) - 1
        __m256 result = _mm256_sub_ps(two_sig, ones);
        
        // Store result
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        output[i] = std::tanh(input[i]);
    }
}

void add_constant_avx2(const float* input, float* output, int size, float constant) {
    // Process 8 elements at a time with AVX2
    int i = 0;
    
    // Broadcast constant to all elements
    __m256 const_vec = _mm256_set1_ps(constant);
    
    for (; i <= size - 8; i += 8) {
        // Load 8 elements from input
        __m256 in = _mm256_loadu_ps(&input[i]);
        
        // Add constant
        __m256 result = _mm256_add_ps(in, const_vec);
        
        // Store result
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        output[i] = input[i] + constant;
    }
}

void multiply_constant_avx2(const float* input, float* output, int size, float constant) {
    // Process 8 elements at a time with AVX2
    int i = 0;
    
    // Broadcast constant to all elements
    __m256 const_vec = _mm256_set1_ps(constant);
    
    for (; i <= size - 8; i += 8) {
        // Load 8 elements from input
        __m256 in = _mm256_loadu_ps(&input[i]);
        
        // Multiply by constant
        __m256 result = _mm256_mul_ps(in, const_vec);
        
        // Store result
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        output[i] = input[i] * constant;
    }
}

void element_wise_operation(const float* input, float* output, int size, ElementWiseOperation op, float constant) {
    // Check if AVX2 is supported
    if (is_avx2_supported() && size >= 8) {
        switch (op) {
            case RELU:
                relu_avx2(input, output, size);
                break;
            case SIGMOID:
                sigmoid_avx2(input, output, size);
                break;
            case TANH:
                tanh_avx2(input, output, size);
                break;
            case ADD:
                add_constant_avx2(input, output, size, constant);
                break;
            case MULTIPLY:
                multiply_constant_avx2(input, output, size, constant);
                break;
            default:
                throw std::runtime_error("Unsupported element-wise operation");
        }
    } else {
        // Fallback to scalar implementation
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
}

void convolution_avx2(
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
) {
    // Process each batch and output channel
    for (int batch = 0; batch < batch_size; batch++) {
        for (int out_c = 0; out_c < output_channels; out_c++) {
            // Process each output pixel
            for (int out_y = 0; out_y < output_height; out_y++) {
                for (int out_x = 0; out_x < output_width; out_x++) {
                    // Calculate input position
                    int in_y_start = out_y * stride_h - padding_h;
                    int in_x_start = out_x * stride_w - padding_w;
                    
                    // Initialize accumulator
                    __m256 sum_vec = _mm256_setzero_ps();
                    float sum_scalar = 0.0f;
                    
                    // Convolve kernel with input at this position
                    for (int ky = 0; ky < kernel_height; ky++) {
                        int in_y = in_y_start + ky;
                        
                        if (in_y >= 0 && in_y < input_height) {
                            for (int kx = 0; kx < kernel_width; kx++) {
                                int in_x = in_x_start + kx;
                                
                                if (in_x >= 0 && in_x < input_width) {
                                    // Process input channels in chunks of 8 using AVX2
                                    int ic = 0;
                                    for (; ic <= input_channels - 8; ic += 8) {
                                        // Load 8 input values
                                        __m256 in_vec = _mm256_loadu_ps(
                                            &input[((batch * input_height + in_y) * input_width + in_x) * input_channels + ic]
                                        );
                                        
                                        // Load 8 kernel values
                                        __m256 kernel_vec = _mm256_loadu_ps(
                                            &kernel[((ky * kernel_width + kx) * input_channels + ic) * output_channels + out_c]
                                        );
                                        
                                        // Multiply and accumulate
                                        sum_vec = _mm256_fmadd_ps(in_vec, kernel_vec, sum_vec);
                                    }
                                    
                                    // Handle remaining channels
                                    for (; ic < input_channels; ic++) {
                                        float in_val = input[((batch * input_height + in_y) * input_width + in_x) * input_channels + ic];
                                        float kernel_val = kernel[((ky * kernel_width + kx) * input_channels + ic) * output_channels + out_c];
                                        sum_scalar += in_val * kernel_val;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Reduce vector sum
                    float sum_array[8];
                    _mm256_storeu_ps(sum_array, sum_vec);
                    
                    float final_sum = sum_scalar;
                    for (int i = 0; i < 8; i++) {
                        final_sum += sum_array[i];
                    }
                    
                    // Store result
                    output[((batch * output_height + out_y) * output_width + out_x) * output_channels + out_c] = final_sum;
                }
            }
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
    int padding_w
) {
    // Check if AVX2 is supported
    if (is_avx2_supported() && input_channels >= 8) {
        convolution_avx2(
            input, kernel, output,
            batch_size, input_height, input_width, input_channels,
            kernel_height, kernel_width, output_channels,
            output_height, output_width,
            stride_h, stride_w, padding_h, padding_w
        );
    } else {
        // Fallback to scalar implementation
        for (int batch = 0; batch < batch_size; batch++) {
            for (int out_c = 0; out_c < output_channels; out_c++) {
                for (int out_y = 0; out_y < output_height; out_y++) {
                    for (int out_x = 0; out_x < output_width; out_x++) {
                        float sum = 0.0f;
                        
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
}

void gf_multiply_avx2(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size
) {
    // Process 16 elements at a time with AVX2
    int i = 0;
    
    for (; i <= size - 16; i += 16) {
        // We need to process each element individually
        // since there's no direct AVX2 support for GF operations
        for (int j = 0; j < 16; j++) {
            uint16_t a_val = a[i + j];
            uint16_t b_val = b[i + j];
            
            if (a_val == 0 || b_val == 0) {
                result[i + j] = 0;
            } else {
                uint16_t log_a = log_table[a_val];
                uint16_t log_b = log_table[b_val];
                uint16_t log_sum = log_a + log_b;
                if (log_sum >= field_size - 1) {
                    log_sum -= field_size - 1;
                }
                result[i + j] = exp_table[log_sum];
            }
        }
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
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

void gf_add_avx2(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size
) {
    // Process 16 elements at a time with AVX2
    int i = 0;
    
    for (; i <= size - 16; i += 16) {
        // Load 16 elements from a and b
        __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        
        // XOR for GF(2^m) addition
        __m256i result_vec = _mm256_xor_si256(a_vec, b_vec);
        
        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        result[i] = a[i] ^ b[i];
    }
}

void gf_multiply(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size
) {
    // Check if AVX2 is supported
    if (is_avx2_supported() && size >= 16) {
        gf_multiply_avx2(a, b, result, size, exp_table, log_table, field_size);
    } else {
        // Fallback to scalar implementation
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
}

void gf_add(
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    int size
) {
    // Check if AVX2 is supported
    if (is_avx2_supported() && size >= 16) {
        gf_add_avx2(a, b, result, size);
    } else {
        // Fallback to scalar implementation
        for (int i = 0; i < size; i++) {
            result[i] = a[i] ^ b[i];
        }
    }
}

void calculate_syndromes_avx2(
    const uint16_t* data,
    int data_length,
    uint16_t* syndromes,
    int syndrome_count,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size
) {
    // Each syndrome requires a separate calculation
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

void calculate_syndromes(
    const uint16_t* data,
    int data_length,
    uint16_t* syndromes,
    int syndrome_count,
    const uint16_t* exp_table,
    const uint16_t* log_table,
    int field_size
) {
    // AVX2 doesn't offer significant advantages for syndrome calculation
    // due to the sequential nature of the algorithm
    calculate_syndromes_avx2(
        data, data_length, syndromes, syndrome_count,
        exp_table, log_table, field_size
    );
}

void polynomial_eval_batch_avx2(
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
) {
    // Process each polynomial and point combination
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
) {
    // AVX2 doesn't offer significant advantages for polynomial evaluation
    // due to the sequential nature of the algorithm and GF operations
    polynomial_eval_batch_avx2(
        polys, poly_lengths, max_poly_len, n_polys,
        points, n_points, results,
        log_table, exp_table, field_size
    );
}

// Helper functions for AVX2 implementations

__m256 exp_approx_avx2(__m256 x) {
    // Approximate exp(x) using a truncated Taylor series
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
    
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 third = _mm256_set1_ps(1.0f/6.0f);
    __m256 fourth = _mm256_set1_ps(1.0f/24.0f);
    __m256 fifth = _mm256_set1_ps(1.0f/120.0f);
    
    // Calculate x²
    __m256 x2 = _mm256_mul_ps(x, x);
    
    // Calculate x³
    __m256 x3 = _mm256_mul_ps(x2, x);
    
    // Calculate x⁴
    __m256 x4 = _mm256_mul_ps(x2, x2);
    
    // Calculate x⁵
    __m256 x5 = _mm256_mul_ps(x4, x);
    
    // Calculate the terms
    __m256 term1 = x;
    __m256 term2 = _mm256_mul_ps(x2, half);
    __m256 term3 = _mm256_mul_ps(x3, third);
    __m256 term4 = _mm256_mul_ps(x4, fourth);
    __m256 term5 = _mm256_mul_ps(x5, fifth);
    
    // Combine the terms: 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
    __m256 result = _mm256_add_ps(one, term1);
    result = _mm256_add_ps(result, term2);
    result = _mm256_add_ps(result, term3);
    result = _mm256_add_ps(result, term4);
    result = _mm256_add_ps(result, term5);
    
    return result;
}

__m256 sigmoid_approx_avx2(__m256 x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_one = _mm256_set1_ps(-1.0f);
    
    // -x
    __m256 neg_x = _mm256_mul_ps(x, neg_one);
    
    // exp(-x)
    __m256 exp_neg_x = exp_approx_avx2(neg_x);
    
    // 1 + exp(-x)
    __m256 denom = _mm256_add_ps(one, exp_neg_x);
    
    // 1 / (1 + exp(-x))
    __m256 result = _mm256_div_ps(one, denom);
    
    return result;
}

} // namespace avx2
} // namespace hardware