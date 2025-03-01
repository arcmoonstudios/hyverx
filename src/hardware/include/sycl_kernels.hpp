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

#ifndef HYVERX_SYCL_KERNELS_HPP
#define HYVERX_SYCL_KERNELS_HPP

// Conditionally include SYCL if the feature is enabled
#if defined(SYCL_ENABLED) || defined(FEATURE_SYCL)
  // Try the standard path first
  #if __has_include(<sycl/sycl.hpp>)
    #include <sycl/sycl.hpp>
  // Try the Intel oneAPI specific path
  #elif __has_include("sycl.hpp")
    #include "sycl.hpp"
  // Finally try the absolute path we know exists on this system
  #else
    #include "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include/sycl/sycl.hpp"
  #endif
#else
// Mock SYCL definitions when SYCL is not available
namespace sycl {
    class queue {};
    class device {};
}
#define SYCL_DISABLED
#endif

#include <array>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <functional>
#include <stdexcept>

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
    /**
     * Compute matrix multiplication C = A * B using tiled algorithm
     * 
     * @param q SYCL queue for execution
     * @param a Pointer to matrix A of size m x k
     * @param b Pointer to matrix B of size k x n
     * @param c Pointer to output matrix C of size m x n
     * @param m Number of rows in A
     * @param k Number of columns in A / rows in B
     * @param n Number of columns in B
     */
    static void compute(sycl::queue& q, 
                        const T* a, 
                        const T* b, 
                        T* c, 
                        int m, int k, int n);
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
    /**
     * Structure representing a sparse matrix in block format
     */
    struct BlockSparseMatrix {
        std::vector<T> values;         // Non-zero block values
        std::vector<int> block_row;    // Block row indices
        std::vector<int> block_col;    // Block column indices
        int num_row_blocks;            // Number of row blocks
        int num_col_blocks;            // Number of column blocks
        int num_nonzero_blocks;        // Number of non-zero blocks
    };

    /**
     * Convert dense matrix to block-sparse format
     * 
     * @param dense Dense matrix data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param threshold Values below this threshold are considered zero
     * @return BlockSparseMatrix representation
     */
    static BlockSparseMatrix denseToBSR(const T* dense, int rows, int cols, T threshold = 1e-6);

    /**
     * Multiply block-sparse matrix with dense matrix
     * 
     * @param q SYCL queue for execution
     * @param bsr Block-sparse matrix A
     * @param dense Dense matrix B
     * @param result Output matrix C
     * @param dense_cols Number of columns in dense matrix B
     */
    static void multiply(sycl::queue& q,
                         const BlockSparseMatrix& bsr,
                         const T* dense,
                         T* result,
                         int dense_cols);
};

/**
 * Galois Field arithmetic operations for error correction
 * 
 * @tparam T Integer type for field elements
 * @tparam FIELD_SIZE Size of the Galois field (typically 2^m - 1)
 */
template <typename T = uint16_t, int FIELD_SIZE = 255>
class GaloisFieldArithmetic {
public:
    /**
     * Initialize log and exp tables for fast GF arithmetic
     * 
     * @param primitive_poly Primitive polynomial defining the field
     * @return Pair of exp and log tables
     */
    static std::pair<std::vector<T>, std::vector<T>> initTables(T primitive_poly);

    /**
     * Multiply two elements in Galois Field
     * 
     * @param a First element
     * @param b Second element
     * @param exp_table Exponential table
     * @param log_table Logarithm table
     * @return Product in GF
     */
    static T multiply(T a, T b, const std::vector<T>& exp_table, const std::vector<T>& log_table);

    /**
     * Add two elements in Galois Field (XOR for GF(2^m))
     * 
     * @param a First element
     * @param b Second element
     * @return Sum in GF
     */
    static T add(T a, T b);

    /**
     * Perform batch operations on arrays in Galois Field
     * 
     * @param q SYCL queue for execution
     * @param a First array
     * @param b Second array
     * @param result Output array
     * @param size Array size
     * @param exp_table Exponential table
     * @param log_table Logarithm table
     * @param operation Operation type (0 for multiply, 1 for add)
     */
    static void batchOperation(sycl::queue& q,
                              const T* a,
                              const T* b,
                              T* result,
                              int size,
                              const std::vector<T>& exp_table,
                              const std::vector<T>& log_table,
                              int operation);
};

/**
 * Reed-Solomon error correction implementation
 * 
 * @tparam T Integer type for field elements
 * @tparam MAX_CODE_LENGTH Maximum length of the codeword
 * @tparam MAX_MESSAGE_LENGTH Maximum length of the message
 */
template <typename T = uint16_t, int MAX_CODE_LENGTH = 255, int MAX_MESSAGE_LENGTH = 223>
class ReedSolomonCode {
public:
    /**
     * Initialize Reed-Solomon encoder/decoder
     * 
     * @param msg_len Message length
     * @param ecc_len Error correction capability (number of parity symbols)
     * @param primitive_poly Primitive polynomial defining the field
     */
    ReedSolomonCode(int msg_len, int ecc_len, T primitive_poly);

    /**
     * Encode a message using Reed-Solomon
     * 
     * @param q SYCL queue for execution
     * @param message Input message
     * @param codeword Output codeword (message + parity)
     */
    void encode(sycl::queue& q, const T* message, T* codeword);

    /**
     * Decode a possibly corrupted codeword
     * 
     * @param q SYCL queue for execution
     * @param received Received codeword with possible errors
     * @param decoded Output decoded message
     * @return Number of errors corrected, or -1 if uncorrectable
     */
    int decode(sycl::queue& q, const T* received, T* decoded);

    /**
     * Inject errors for testing purposes
     * 
     * @param original Original codeword
     * @param error_positions Positions of errors to inject
     * @return Corrupted codeword
     */
    std::vector<T> injectErrors(const std::vector<T>& original, const std::vector<int>& error_positions);

private:
    int message_length;
    int ecc_length;
    int code_length;
    std::vector<T> generator_poly;
    std::vector<T> exp_table;
    std::vector<T> log_table;

    /**
     * Calculate syndromes for error detection
     * 
     * @param q SYCL queue for execution
     * @param received Received codeword
     * @return Vector of syndromes
     */
    std::vector<T> calculateSyndromes(sycl::queue& q, const T* received);
    
    /**
     * Find error locator polynomial using Berlekamp-Massey algorithm
     * 
     * @param syndromes Syndrome values
     * @return Error locator polynomial coefficients
     */
    std::vector<T> findErrorLocator(const std::vector<T>& syndromes);
    
    /**
     * Find roots of the error locator polynomial (Chien search)
     * 
     * @param q SYCL queue for execution
     * @param error_locator Error locator polynomial
     * @return Error positions
     */
    std::vector<int> findErrorPositions(sycl::queue& q, const std::vector<T>& error_locator);
    
    /**
     * Calculate error values at error positions (Forney algorithm)
     * 
     * @param syndromes Syndrome values
     * @param error_locator Error locator polynomial
     * @param error_positions Positions of errors
     * @return Error values
     */
    std::vector<T> findErrorValues(const std::vector<T>& syndromes, 
                                  const std::vector<T>& error_locator,
                                  const std::vector<int>& error_positions);
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
    
    /**
     * Get capabilities of available devices
     * 
     * @param devices Vector of SYCL devices
     * @return Vector of device capabilities
     */
    std::vector<DeviceCapability> getDeviceCapabilities(const std::vector<sycl::device>& devices);
    
public:
    /**
     * Distribute workload across available devices based on their capabilities
     * 
     * @param kernel_func Function to execute on each device
     * @param data Data to process
     * @param total_elements Total number of elements to process
     * @param force_gpu Whether to force execution on GPU only
     */
    void distributeWorkload(
        std::function<void(sycl::queue&, T*, int, int)> kernel_func,
        T* data,
        int total_elements,
        bool force_gpu = false);
};

} // namespace hybrid_acceleration

#endif // HYVERX_SYCL_KERNELS_HPP