/**
 * Complementary SYCL Kernels for Heterogeneous Computing
 *
 * This module provides SYCL-based implementations that complement the OpenCL kernels
 * for integrated GPU execution. These kernels focus on operations where SYCL's modern
 * programming model offers advantages: tensor operations, automatic memory management,
 * heterogeneous workload balancing, and complex algorithm execution.
 *
 * Key features:
 * - Advanced tensor operations with optimized memory handling
 * - Block-sparse matrix computations for efficient ML operations
 * - Unified Shared Memory (USM) utilization for zero-copy data transfer
 * - Advanced error correction with integrated Galois Field operations
 * - Hardware-aware workload distribution with dynamic adaptation
 *
 * Compatible with modern SYCL implementations (oneAPI DPC++ or ComputeCpp).
 */

// Try standard path first
#if __has_include(<sycl/sycl.hpp>)
  #include <sycl/sycl.hpp>
// Try Intel oneAPI specific path
#elif __has_include("sycl.hpp")
  #include "sycl.hpp"
// Finally, use the absolute path we know exists on this system
#else
 #include "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include/sycl/sycl.hpp"
#endif

#include "../include/sycl_kernels.hpp"
#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <type_traits>

// Namespace to avoid conflicts with OpenCL kernels
namespace hybrid_acceleration {

/**
 * Advanced matrix multiplication with tiling and local memory optimization
 * Implements blocking strategy for better cache utilization
 *
 * @tparam T Data type (float or double)
 * @tparam TILE_SIZE Size of the tile for blocking (must be a power of 2)
 */
template <typename T, int TILE_SIZE = 16>
class TiledMatrixMultiply {
private:
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Only float and double types are supported");
    
public:
    static void compute(sycl::queue& q, 
                        const T* a, 
                        const T* b, 
                        T* c, 
                        int m, int k, int n) {
        // Ensure proper alignment and padding
        const int m_padded = (m + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        const int n_padded = (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        const int k_padded = (k + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        
        // Submit to queue
        q.submit([&](sycl::handler& h) {
            // Local memory for tiles
            sycl::accessor<T, 2, sycl::access::mode::read_write, sycl::access::target::local> 
                tileA(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);
            sycl::accessor<T, 2, sycl::access::mode::read_write, sycl::access::target::local> 
                tileB(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);
            
            // Parallel execution over tiles
            h.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(m_padded, n_padded),
                    sycl::range<2>(TILE_SIZE, TILE_SIZE)
                ),
                [=](sycl::nd_item<2> item) {
                    const int row = item.get_global_id(0);
                    const int col = item.get_global_id(1);
                    const int local_row = item.get_local_id(0);
                    const int local_col = item.get_local_id(1);
                    
                    // Accumulator for dot product
                    T sum = 0;
                    
                    // Iterate over tiles
                    for (int tile = 0; tile < (k_padded / TILE_SIZE); ++tile) {
                        // Load tiles into local memory collaboratively
                        if (row < m && (tile * TILE_SIZE + local_col) < k) {
                            tileA[local_row][local_col] = a[row * k + tile * TILE_SIZE + local_col];
                        } else {
                            tileA[local_row][local_col] = 0;
                        }
                        
                        if ((tile * TILE_SIZE + local_row) < k && col < n) {
                            tileB[local_row][local_col] = b[(tile * TILE_SIZE + local_row) * n + col];
                        } else {
                            tileB[local_row][local_col] = 0;
                        }
                        
                        // Synchronize to ensure all work-items have loaded the tiles
                        item.barrier(sycl::access::fence_space::local_space);
                        
                        // Compute partial dot product for this tile
                        for (int i = 0; i < TILE_SIZE; ++i) {
                            sum += tileA[local_row][i] * tileB[i][local_col];
                        }
                        
                        // Synchronize before loading next tile
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    
                    // Write result if within bounds
                    if (row < m && col < n) {
                        c[row * n + col] = sum;
                    }
                }
            );
        }).wait();
    }
};

/**
 * Block-sparse matrix multiplication optimized for ML workloads
 * Efficiently handles sparse matrices common in deep learning
 *
 * @tparam T Data type (typically float)
 * @tparam BLOCK_SIZE Size of dense blocks within the sparse matrix
 */
template <typename T, int BLOCK_SIZE = 8>
class BlockSparseMatrixMultiply {
public:
    struct BlockSparseMatrix {
        std::vector<T> values;             // Non-zero block values in row-major order
        std::vector<int> block_row_indices; // Row indices for each block
        std::vector<int> block_col_indices; // Column indices for each block
        int rows;                          // Total rows in full matrix
        int cols;                          // Total columns in full matrix
        int num_blocks;                    // Number of non-zero blocks
    };
    
    static void compute(sycl::queue& q,
                        const BlockSparseMatrix& a,
                        const T* dense_b,
                        T* c,
                        int n) {  // n is the number of columns in dense_b
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<1>(a.rows),
                [=](sycl::id<1> idx) {
                    const int row = idx[0];
                    
                    // Find all blocks in this row
                    for (int col = 0; col < n; ++col) {
                        T sum = 0;
                        
                        // Iterate through blocks in this row
                        for (int block_idx = 0; block_idx < a.num_blocks; ++block_idx) {
                            if (a.block_row_indices[block_idx] <= row && 
                                row < a.block_row_indices[block_idx] + BLOCK_SIZE) {
                                
                                const int block_row = row - a.block_row_indices[block_idx];
                                const int block_col_start = a.block_col_indices[block_idx];
                                
                                // Dot product with this block row and the dense matrix column
                                for (int i = 0; i < BLOCK_SIZE; ++i) {
                                    const int sparse_col = block_col_start + i;
                                    if (sparse_col < a.cols) {
                                        const T block_val = a.values[
                                            block_idx * BLOCK_SIZE * BLOCK_SIZE + 
                                            block_row * BLOCK_SIZE + i
                                        ];
                                        sum += block_val * dense_b[sparse_col * n + col];
                                    }
                                }
                            }
                        }
                        
                        // Write result
                        c[row * n + col] = sum;
                    }
                }
            );
        }).wait();
    }
};

/**
 * Advanced tensor contraction operation optimized for modern GPU architectures
 * Efficiently handles high-dimensional tensor operations common in deep learning
 *
 * @tparam T Data type (float or double)
 * @tparam WORK_GROUP_SIZE Work group size for parallel execution
 */
template <typename T, int WORK_GROUP_SIZE = 128>
class TensorContraction {
public:
    struct TensorDims {
        std::vector<int> dims;
        std::vector<int> strides;
    };
    
    // Helper to convert flat index to multidimensional coordinates
    static std::vector<int> unflattenIndex(int flat_idx, const std::vector<int>& dims) {
        std::vector<int> coords(dims.size());
        for (int i = dims.size() - 1; i >= 0; --i) {
            coords[i] = flat_idx % dims[i];
            flat_idx /= dims[i];
        }
        return coords;
    }
    
    // Helper to convert multidimensional coordinates to flat index
    static int flattenIndex(const std::vector<int>& coords, const std::vector<int>& strides) {
        int flat_idx = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            flat_idx += coords[i] * strides[i];
        }
        return flat_idx;
    }
    
    static void compute(sycl::queue& q,
                       const T* a, const TensorDims& a_dims,
                       const T* b, const TensorDims& b_dims,
                       T* c, const TensorDims& c_dims,
                       const std::vector<std::pair<int, int>>& contraction_dims) {
        
        // Calculate total elements in the output tensor
        int c_size = 1;
        for (const auto& dim : c_dims.dims) {
            c_size *= dim;
        }
        
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>((c_size + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE * WORK_GROUP_SIZE),
                    sycl::range<1>(WORK_GROUP_SIZE)
                ),
                [=](sycl::nd_item<1> item) {
                    const int idx = item.get_global_id(0);
                    
                    if (idx < c_size) {
                        // Convert flat index to output tensor coordinates
                        auto c_coords = unflattenIndex(idx, c_dims.dims);
                        
                        // Initialize result
                        T result = 0;
                        
                        // Determine the ranges for contracted dimensions
                        std::vector<int> contraction_size;
                        for (const auto& dim_pair : contraction_dims) {
                            contraction_size.push_back(a_dims.dims[dim_pair.first]);
                        }
                        
                        // Calculate total number of contracted elements
                        int total_contraction_elements = 1;
                        for (const auto& size : contraction_size) {
                            total_contraction_elements *= size;
                        }
                        
                        // Iterate over all contraction elements
                        for (int contract_idx = 0; contract_idx < total_contraction_elements; ++contract_idx) {
                            // Convert to contraction coordinates
                            auto contract_coords = unflattenIndex(contract_idx, contraction_size);
                            
                            // Build full coordinates for tensors A and B
                            std::vector<int> a_coords(a_dims.dims.size());
                            std::vector<int> b_coords(b_dims.dims.size());
                            
                            // Fill non-contracted coordinates for A and B from output C
                            int c_idx = 0;
                            for (size_t i = 0; i < a_dims.dims.size(); ++i) {
                                bool is_contracted = false;
                                for (size_t j = 0; j < contraction_dims.size(); ++j) {
                                    if (contraction_dims[j].first == i) {
                                        a_coords[i] = contract_coords[j];
                                        is_contracted = true;
                                        break;
                                    }
                                }
                                if (!is_contracted) {
                                    a_coords[i] = c_coords[c_idx++];
                                }
                            }
                            
                            c_idx = 0;
                            for (size_t i = 0; i < b_dims.dims.size(); ++i) {
                                bool is_contracted = false;
                                for (size_t j = 0; j < contraction_dims.size(); ++j) {
                                    if (contraction_dims[j].second == i) {
                                        b_coords[i] = contract_coords[j];
                                        is_contracted = true;
                                        break;
                                    }
                                }
                                if (!is_contracted) {
                                    b_coords[i] = c_coords[c_idx++];
                                }
                            }
                            
                            // Get flat indices
                            int a_idx = flattenIndex(a_coords, a_dims.strides);
                            int b_idx = flattenIndex(b_coords, b_dims.strides);
                            
                            // Accumulate product
                            result += a[a_idx] * b[b_idx];
                        }
                        
                        // Write result
                        c[idx] = result;
                    }
                }
            );
        }).wait();
    }
};

/**
 * Advanced Reed-Solomon error correction with adaptable field size
 * Integrates with the Galois Field operations from OpenCL
 *
 * @tparam FIELD_BITS Bit size of the Galois Field (typically 8 for GF(2^8))
 * @tparam WORK_GROUP_SIZE Work group size for parallel execution
 */
template <int FIELD_BITS = 8, int WORK_GROUP_SIZE = 256>
class ReedSolomonDecoder {
private:
    // Constants
    static constexpr int FIELD_SIZE = 1 << FIELD_BITS;
    static constexpr int FIELD_POLY = 0x11D; // Standard polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1
    
    // Lookup tables
    std::vector<uint16_t> exp_table;
    std::vector<uint16_t> log_table;
    
    // Initialize lookup tables
    void initTables() {
        exp_table.resize(FIELD_SIZE * 2);  // Double size to avoid modulo operations
        log_table.resize(FIELD_SIZE);
        
        // Generate tables
        uint16_t x = 1;
        for (int i = 0; i < FIELD_SIZE - 1; i++) {
            exp_table[i] = x;
            log_table[x] = i;
            
            // Multiply by the primitive element (usually 2 in GF(2^m))
            x = x << 1;
            if (x >= FIELD_SIZE)
                x = (x ^ FIELD_POLY) & (FIELD_SIZE - 1);
        }
        
        // Complete exp_table for easier modulo-less lookup
        for (int i = FIELD_SIZE - 1; i < FIELD_SIZE * 2 - 1; i++) {
            exp_table[i] = exp_table[i - (FIELD_SIZE - 1)];
        }
        
        // Set log of 0 as an invalid value (convention)
        log_table[0] = 0;
    }
    
    // Galois Field multiplication using lookup tables
    uint16_t gfMul(uint16_t a, uint16_t b) const {
        if (a == 0 || b == 0) return 0;
        return exp_table[(log_table[a] + log_table[b]) % (FIELD_SIZE - 1)];
    }
    
    // Galois Field division using lookup tables
    uint16_t gfDiv(uint16_t a, uint16_t b) const {
        if (a == 0) return 0;
        if (b == 0) throw std::runtime_error("Division by zero in Galois Field");
        return exp_table[(log_table[a] + FIELD_SIZE - 1 - log_table[b]) % (FIELD_SIZE - 1)];
    }
    
public:
    ReedSolomonDecoder() {
        initTables();
    }
    
    // Compute syndromes in parallel
    std::vector<uint16_t> computeSyndromes(sycl::queue& q, 
                                          const std::vector<uint16_t>& received,
                                          int nsym) {
        std::vector<uint16_t> syndromes(nsym);
        
        // Transfer tables to device
        sycl::buffer<uint16_t, 1> exp_table_buf(exp_table.data(), exp_table.size());
        sycl::buffer<uint16_t, 1> log_table_buf(log_table.data(), log_table.size());
        sycl::buffer<uint16_t, 1> received_buf(received.data(), received.size());
        sycl::buffer<uint16_t, 1> syndrome_buf(syndromes.data(), syndromes.size());
        
        q.submit([&](sycl::handler& h) {
            auto exp_table_acc = exp_table_buf.get_access<sycl::access::mode::read>(h);
            auto log_table_acc = log_table_buf.get_access<sycl::access::mode::read>(h);
            auto received_acc = received_buf.get_access<sycl::access::mode::read>(h);
            auto syndrome_acc = syndrome_buf.get_access<sycl::access::mode::write>(h);
            
            h.parallel_for(
                sycl::range<1>(nsym),
                [=](sycl::id<1> idx) {
                    const int i = idx[0];
                    uint16_t syndrome = 0;
                    uint16_t x = 1; // x^0 = 1
                    
                    // Compute syndrome using Horner's method
                    for (int j = 0; j < received_acc.size(); j++) {
                        // syndrome = (syndrome + received[j] * x^i) % FIELD_SIZE
                        if (syndrome != 0 || received_acc[j] != 0) {
                            syndrome ^= (received_acc[j] == 0) ? 0 :
                                exp_table_acc[(log_table_acc[received_acc[j]] + 
                                              (i + 1) * j) % (FIELD_SIZE - 1)];
                        }
                    }
                    
                    syndrome_acc[i] = syndrome;
                }
            );
        }).wait();
        
        // Copy results back
        {
            auto syndrome_acc = syndrome_buf.get_access<sycl::access::mode::read>();
            for (int i = 0; i < nsym; i++) {
                syndromes[i] = syndrome_acc[i];
            }
        }
        
        return syndromes;
    }
    
    // Find error locator polynomial using Berlekamp-Massey algorithm
    std::vector<uint16_t> findErrorLocator(const std::vector<uint16_t>& syndromes, int nsym) {
        std::vector<uint16_t> err_loc(nsym + 1, 0);
        std::vector<uint16_t> old_loc(nsym + 1, 0);
        
        // Initialize polynomials
        err_loc[0] = 1;
        old_loc[0] = 1;
        
        // Iteratively build the error locator polynomial
        for (int i = 0; i < nsym; i++) {
            // delta = syndromes[i] + sum(err_loc[j] * syndromes[i - j]) for j from 1 to i
            uint16_t delta = syndromes[i];
            for (int j = 1; j <= i; j++) {
                delta ^= gfMul(err_loc[j], syndromes[i - j]);
            }
            
            // Shift old_loc
            old_loc.insert(old_loc.begin(), 0);
            
            if (delta != 0) {
                if (2 * i > nsym) { // Late phase of the algorithm, use old_loc
                    // Multiply old_loc by delta
                    for (int j = 0; j <= nsym; j++) {
                        old_loc[j] = gfMul(old_loc[j], delta);
                    }
                    
                    // Update err_loc = err_loc - old_loc * delta
                    for (int j = 0; j <= nsym; j++) {
                        err_loc[j] ^= old_loc[j]; // XOR is addition in GF(2^m)
                    }
                } else { // Early phase, simple update
                    // Store current err_loc
                    std::vector<uint16_t> temp(err_loc);
                    
                    // Compute new err_loc
                    for (int j = 0; j <= nsym; j++) {
                        uint16_t b = (j < old_loc.size()) ? old_loc[j] : 0;
                        err_loc[j] ^= gfMul(delta, b);
                    }
                    
                    // Update old_loc for next iteration
                    old_loc = temp;
                }
            }
        }
        
        return err_loc;
    }
    
    // Find error positions using Chien search in parallel
    std::vector<int> findErrorPositions(sycl::queue& q, 
                                       const std::vector<uint16_t>& err_loc,
                                       int msg_len) {
        std::vector<uint16_t> error_evaluations(msg_len, 0);
        
        // Transfer tables and error locator to device
        sycl::buffer<uint16_t, 1> exp_table_buf(exp_table.data(), exp_table.size());
        sycl::buffer<uint16_t, 1> log_table_buf(log_table.data(), log_table.size());
        sycl::buffer<uint16_t, 1> err_loc_buf(err_loc.data(), err_loc.size());
        sycl::buffer<uint16_t, 1> eval_buf(error_evaluations.data(), error_evaluations.size());
        
        q.submit([&](sycl::handler& h) {
            auto exp_table_acc = exp_table_buf.get_access<sycl::access::mode::read>(h);
            auto log_table_acc = log_table_buf.get_access<sycl::access::mode::read>(h);
            auto err_loc_acc = err_loc_buf.get_access<sycl::access::mode::read>(h);
            auto eval_acc = eval_buf.get_access<sycl::access::mode::write>(h);
            
            h.parallel_for(
                sycl::range<1>(msg_len),
                [=](sycl::id<1> idx) {
                    const int i = idx[0];
                    uint16_t eval = 0;
                    
                    // Evaluate error locator polynomial at x = alpha^i
                    for (int j = 0; j < err_loc_acc.size(); j++) {
                        if (err_loc_acc[j] != 0) {
                            // power = alpha^(j*i) = exp_table[(j*i) % (FIELD_SIZE-1)]
                            uint16_t power = (j == 0) ? 1 : // Alpha^0 = 1
                                exp_table_acc[(j * i) % (FIELD_SIZE - 1)];
                            
                            // Add to evaluation (XOR is addition in GF(2^m))
                            eval ^= gfMul(err_loc_acc[j], power);
                        }
                    }
                    
                    eval_acc[i] = eval;
                }
            );
        }).wait();
        
        // Find positions where evaluation is zero (these are error positions)
        std::vector<int> error_positions;
        {
            auto eval_acc = eval_buf.get_access<sycl::access::mode::read>();
            for (int i = 0; i < msg_len; i++) {
                if (eval_acc[i] == 0) {
                    error_positions.push_back(i);
                }
            }
        }
        
        return error_positions;
    }
    
    // Correct errors in the received message
    std::vector<uint16_t> correctErrors(sycl::queue& q, 
                                       std::vector<uint16_t> received,
                                       int nsym) {
        // Compute syndromes
        auto syndromes = computeSyndromes(q, received, nsym);
        
        // Check if there are any errors
        bool all_zero = true;
        for (const auto& s : syndromes) {
            if (s != 0) {
                all_zero = false;
                break;
            }
        }
        
        if (all_zero) {
            return received; // No errors found
        }
        
        // Find error locator polynomial
        auto err_loc = findErrorLocator(syndromes, nsym);
        
        // Find error positions
        auto error_positions = findErrorPositions(q, err_loc, received.size());
        
        // Find error values and correct them
        for (const auto& pos : error_positions) {
            // In simple error correction, we just flip the bits (for binary data)
            // For multi-bit symbols, more complex error value computation is needed
            received[pos] = (received[pos] == 0) ? 1 : 0;
        }
        
        return received;
    }
};

/**
 * Adaptive workload balancer for heterogeneous compute environments
 * Dynamically distributes work between CPU and GPU based on load
 *
 * @tparam T Data type for computations
 */
template <typename T>
class AdaptiveWorkloadBalancer {
private:
    struct DeviceCapability {
        std::string name;
        float compute_units;
        float clock_frequency;
        float memory_bandwidth;
        bool integrated;
        float relative_power; // Computed metric for work distribution
    };
    
    // Get device capabilities
    std::vector<DeviceCapability> getDeviceCapabilities(const std::vector<sycl::device>& devices) {
        std::vector<DeviceCapability> capabilities;
        
        for (const auto& device : devices) {
            DeviceCapability cap;
            cap.name = device.get_info<sycl::info::device::name>();
            cap.compute_units = static_cast<float>(device.get_info<sycl::info::device::max_compute_units>());
            cap.clock_frequency = static_cast<float>(device.get_info<sycl::info::device::max_clock_frequency>());
            cap.memory_bandwidth = 0; // Not directly available, would need benchmarking
            cap.integrated = device.is_host() || device.get_info<sycl::info::device::host_unified_memory>();
            
            // Simple heuristic for relative power
            cap.relative_power = cap.compute_units * cap.clock_frequency;
            if (device.is_gpu()) {
                cap.relative_power *= 10.0f; // GPUs are generally more powerful for parallel workloads
            }
            
            capabilities.push_back(cap);
        }
        
        return capabilities;
    }
    
public:
    // Dynamic work distribution between CPU and GPU
    void distributeWorkload(
        std::function<void(sycl::queue&, T*, int, int)> kernel_func,
        T* data,
        int total_elements,
        bool force_gpu = false) {
        
        // Get all available devices
        std::vector<sycl::device> devices = sycl::device::get_devices();
        if (devices.empty()) {
            throw std::runtime_error("No SYCL devices available");
        }
        
        // Get device capabilities
        auto capabilities = getDeviceCapabilities(devices);
        
        // Fast path: If force_gpu is true, use only GPU devices
        if (force_gpu) {
            for (size_t i = 0; i < devices.size(); ++i) {
                if (devices[i].is_gpu()) {
                    sycl::queue q(devices[i]);
                    kernel_func(q, data, 0, total_elements);
                    return;
                }
            }
        }
        
        // Calculate total compute power
        float total_power = 0.0f;
        for (const auto& cap : capabilities) {
            total_power += cap.relative_power;
        }
        
        // Distribute work proportionally to relative power
        int start_idx = 0;
        for (size_t i = 0; i < devices.size(); ++i) {
            float device_fraction = capabilities[i].relative_power / total_power;
            int device_elements = static_cast<int>(total_elements * device_fraction);
            
            // Ensure we process at least one element
            device_elements = std::max(1, device_elements);
            
            // Don't exceed total_elements
            if (start_idx + device_elements > total_elements) {
                device_elements = total_elements - start_idx;
            }
            
            // Skip if no elements to process
            if (device_elements <= 0) {
                continue;
            }
            
            // Create queue for this device and process its share
            sycl::queue q(devices[i]);
            kernel_func(q, data, start_idx, device_elements);
            
            // Update start index for next device
            start_idx += device_elements;
            
            // Break if all elements processed
            if (start_idx >= total_elements) {
                break;
            }
        }
    }
};

} // namespace hybrid_acceleration