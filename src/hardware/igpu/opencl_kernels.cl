/**
 * HyVERX Hardware Acceleration OpenCL Kernels
 * 
 * This module provides a comprehensive set of optimized kernels for both
 * neural network computation and Galois field operations, primarily targeting
 * hardware acceleration for error correction codes and machine learning tasks.
 * 
 * Key features:
 * - Neural network primitives (matrix multiplication, activation functions)
 * - Galois field arithmetic optimized for Reed-Solomon codes
 * - Polynomial evaluation with batching support
 * - Syndrome calculation for error correction
 * 
 * All kernels are optimized for parallel execution and memory access patterns.
 */

/**
 * Matrix multiplication kernel optimized for cache coherence
 * Computes C = A * B where A is (m x k) and B is (k x n)
 *
 * @param a Input matrix A (m x k)
 * @param b Input matrix B (k x n)
 * @param c Output matrix C (m x n)
 * @param m Number of rows in A
 * @param k Number of columns in A / rows in B
 * @param n Number of columns in B
 */
__kernel void matrix_multiply(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int m,
    const int k,
    const int n
) 
{
    // Get global position in grid
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // Boundary check
    if (row < m && col < n) {
        float sum = 0.0f;
        // Loop unrolling could be applied here for further optimization
        // on hardware that supports it
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

/**
 * ReLU activation function
 * Computes f(x) = max(0, x) element-wise
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param size Number of elements
 */
__kernel void relu_kernel(
    __global const float* input,
    __global float* output,
    const int size
) {
    const int id = get_global_id(0);
    if (id < size) {
        output[id] = max(0.0f, input[id]);
    }
}

/**
 * Sigmoid activation function
 * Computes f(x) = 1 / (1 + exp(-x)) element-wise
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param size Number of elements
 */
__kernel void sigmoid_kernel(
    __global const float* input,
    __global float* output,
    const int size
) {
    const int id = get_global_id(0);
    if (id < size) {
        output[id] = 1.0f / (1.0f + exp(-input[id]));
    }
}

/**
 * Tanh activation function
 * Computes f(x) = tanh(x) element-wise
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param size Number of elements
 */
__kernel void tanh_kernel(
    __global const float* input,
    __global float* output,
    const int size
) {
    const int id = get_global_id(0);
    if (id < size) {
        output[id] = tanh(input[id]);
    }
}

/**
 * Element-wise addition with a constant
 * Computes output = input + constant
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param size Number of elements
 * @param constant Value to add to each element
 */
__kernel void add_constant_kernel(
    __global const float* input,
    __global float* output,
    const int size,
    const float constant
) {
    const int id = get_global_id(0);
    if (id < size) {
        output[id] = input[id] + constant;
    }
}

/**
 * Element-wise multiplication with a constant
 * Computes output = input * constant
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param size Number of elements
 * @param constant Value to multiply each element by
 */
__kernel void multiply_constant_kernel(
    __global const float* input,
    __global float* output,
    const int size,
    const float constant
) {
    const int id = get_global_id(0);
    if (id < size) {
        output[id] = input[id] * constant;
    }
}

/**
 * Galois field multiplication operation
 * Uses precomputed log and exponential tables for efficient computation
 *
 * @param a First input array
 * @param b Second input array
 * @param result Output array
 * @param size Number of elements
 * @param exp_table Precomputed exponential table
 * @param log_table Precomputed logarithm table
 * @param field_size Size of the Galois field (typically 2^m - 1)
 */
__kernel void gf_multiply(
    __global const ushort* a,
    __global const ushort* b,
    __global ushort* result,
    const int size,
    __global const ushort* exp_table,
    __global const ushort* log_table,
    const int field_size
) {
    const int id = get_global_id(0);
    if (id < size) {
        const ushort a_val = a[id];
        const ushort b_val = b[id];
        
        if (a_val == 0 || b_val == 0) {
            result[id] = 0;
        } else {
            const ushort log_a = log_table[a_val];
            const ushort log_b = log_table[b_val];
            const ushort log_sum = (log_a + log_b);
            const ushort mod_sum = log_sum >= (field_size - 1) ? log_sum - (field_size - 1) : log_sum;
            result[id] = exp_table[mod_sum];
        }
    }
}

/**
 * Galois field addition operation (XOR)
 * In GF(2^m), addition is equivalent to XOR
 *
 * @param a First input array
 * @param b Second input array
 * @param result Output array
 * @param size Number of elements
 */
__kernel void gf_add(
    __global const ushort* a,
    __global const ushort* b,
    __global ushort* result,
    const int size
) {
    const int id = get_global_id(0);
    if (id < size) {
        result[id] = a[id] ^ b[id]; // XOR for GF(2^m) addition
    }
}

/**
 * Syndrome calculation for Reed-Solomon codes
 * Optimized implementation using logarithm and exponential tables
 *
 * @param data Input data array
 * @param data_length Length of input data
 * @param syndromes Output syndromes array
 * @param syndrome_count Number of syndromes to calculate
 * @param exp_table Precomputed exponential table
 * @param log_table Precomputed logarithm table
 * @param field_size Size of the Galois field
 */
__kernel void calculate_syndromes(
    __global const ushort* data,
    const int data_length,
    __global ushort* syndromes,
    const int syndrome_count,
    __global const ushort* exp_table,
    __global const ushort* log_table,
    const int field_size
) 
{
    const int syndrome_idx = get_global_id(0);
    
    if (syndrome_idx < syndrome_count) {
        const ushort alpha_i = syndrome_idx + 1;
        ushort sum = 0;
        ushort alpha_power = 1; // α^0 = 1
        
        // Optimized Horner's method for polynomial evaluation
        for (int j = 0; j < data_length; j++) {
            // Compute data[j] * alpha_power and add to sum
            if (data[j] != 0 && alpha_power != 0) {
                ushort log_sum = (log_table[data[j]] + log_table[alpha_power]) % field_size;
                sum ^= exp_table[log_sum];
            }
            
            // Update alpha_power to alpha_power * alpha_i for next iteration
            if (alpha_power != 0) {
                alpha_power = exp_table[(log_table[alpha_power] + log_table[alpha_i]) % field_size];
            }
        }
        
        syndromes[syndrome_idx] = sum;
    }
}

/**
 * Batch polynomial evaluation in Galois field
 * Uses Horner's method for efficient evaluation
 *
 * @param polys Array of polynomial coefficients
 * @param poly_lengths Array of polynomial lengths
 * @param max_poly_len Maximum polynomial length
 * @param n_polys Number of polynomials
 * @param points Array of evaluation points
 * @param n_points Number of evaluation points
 * @param results Output array for evaluation results
 * @param exp_table Precomputed exponential table
 * @param log_table Precomputed logarithm table
 * @param field_size Size of the Galois field
 */
__kernel void polynomial_eval_batch(
    __global const ushort* polys,
    __global const int* poly_lengths,
    const int max_poly_len,
    const int n_polys,
    __global const ushort* points,
    const int n_points,
    __global ushort* results,
    __global const ushort* log_table,
    __global const ushort* exp_table,
    const int field_size
) 
{
    const int poly_idx = get_global_id(1);
    const int point_idx = get_global_id(0);
    
    if (point_idx < n_points && poly_idx < n_polys) {
        const ushort point = points[point_idx];
        const int poly_len = poly_lengths[poly_idx];
        const int poly_offset = poly_idx * max_poly_len;
        
        // Use Horner's method for polynomial evaluation
        ushort result = 0;
        
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
                        const ushort log_sum = log_table[result] + log_table[point];
                        const ushort mod_sum = log_sum >= (field_size - 1) ? log_sum - (field_size - 1) : log_sum;
                        result = exp_table[mod_sum];
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