//! Hardware acceleration infrastructure for heterogeneous computing systems.
//!
//! This module provides a comprehensive abstraction layer for hardware acceleration
//! across diverse computational architectures, including CPU (AVX2/OpenMP),
//! discrete GPUs (CUDA/Tensor Cores), and integrated GPUs (SYCL/OpenCL).
//!
//! The architecture is designed around a trait-based polymorphic interface that
//! enables transparent acceleration selection, automatic fallback mechanisms,
//! and dynamic hardware capability discovery. Primary operations include matrix
//! manipulations, neural network primitives, and Galois field arithmetic for
//! error correction codes.
//!
//! Key components:
//!
//! * **Hardware Abstraction Layer**: Unified interface across heterogeneous computing platforms
//! * **Dynamic Dispatching**: Runtime selection of optimal acceleration strategy
//! * **Composite Accelerators**: Intelligent workload distribution across available devices
//! * **Tensor Operations**: Specialized implementations for neural network computation
//! * **ECC Operations**: Galois field arithmetic for error correction codes

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;

pub mod cpu;
pub mod dgpu;
pub mod igpu;

// Re-export accelerator registration functions
pub use cpu::register_cpu_accelerators;
pub use dgpu::register_cuda_accelerators;
pub use igpu::register_opencl_accelerators;

use crate::error::{Error, Result};
use crate::galois::GaloisField;
use crate::xypher_grid::XypherGrid;

/// Types of hardware accelerators available in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorType {
    /// CPU with AVX2/OpenMP SIMD acceleration
    Cpu,
    /// NVIDIA GPUs with CUDA/Tensor Cores
    Cuda,
    /// Cross-platform GPU acceleration with OpenCL/SYCL
    OpenCL,
    /// Composite accelerator using multiple physical devices
    Composite,
}

impl std::fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceleratorType::Cpu => write!(f, "CPU"),
            AcceleratorType::Cuda => write!(f, "CUDA"),
            AcceleratorType::OpenCL => write!(f, "OpenCL"),
            AcceleratorType::Composite => write!(f, "Composite"),
        }
    }
}

/// Hardware capabilities profile for a specific accelerator implementation.
///
/// This structure encapsulates the capabilities and resources available
/// to a particular hardware accelerator, enabling intelligent workload
/// distribution and fallback mechanisms.
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Whether AVX2 SIMD extensions are available
    pub avx2_available: bool,
    /// Whether OpenMP parallel processing is available
    pub openmp_available: bool,
    /// Whether CUDA acceleration is available
    pub cuda_available: bool,
    /// Number of CUDA devices available
    pub cuda_device_count: usize,
    /// Whether OpenCL acceleration is available
    pub opencl_available: bool,
    /// Number of OpenCL devices available
    pub opencl_device_count: usize,
    /// Number of processing cores/units available
    pub processor_count: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Whether Hamming code acceleration is available
    pub hamming_available: bool,
    /// Whether BCH code acceleration is available
    pub bch_available: bool,
    /// Whether Reed-Muller code acceleration is available
    pub reed_muller_available: bool,
    /// Whether Turbo code acceleration is available
    pub turbo_available: bool,
    /// Whether Convolutional code acceleration is available
    pub convolutional_available: bool,
    /// Whether Fountain code acceleration is available
    pub fountain_available: bool,
}

/// Performance statistics for hardware acceleration operations.
///
/// This structure tracks performance metrics for hardware accelerators,
/// enabling performance profiling, bottleneck identification, and
/// adaptive optimization strategies.
#[derive(Debug, Clone, Default)]
pub struct HardwareStatistics {
    /// Number of synchronous operations performed
    pub sync_operations: usize,
    /// Number of asynchronous operations performed
    pub async_operations: usize,
    /// Number of CPU operations performed
    pub cpu_operations: usize,
    /// Number of CUDA operations performed
    pub cuda_operations: usize,
    /// Number of OpenCL operations performed
    pub opencl_operations: usize,
    /// Number of AVX2 operations performed
    pub avx2_operations: usize,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
}

/// Tensor operations for neural-symbolic integration.
///
/// This enumeration defines the various tensor operations supported by
/// the hardware acceleration infrastructure, facilitating efficient
/// neural network computations and neural-symbolic integration.
#[derive(Debug, Clone)]
pub enum TensorOperation {
    /// Matrix multiplication operation: C = A * B
    MatrixMultiply {
        /// First matrix (m x k)
        a: Vec<f32>,
        /// Second matrix (k x n)
        b: Vec<f32>,
        /// Result matrix (m x n)
        c: Arc<parking_lot::Mutex<Vec<f32>>>,
        /// Dimensions (m, k, n)
        dims: (usize, usize, usize),
    },
    /// Element-wise operation on a tensor
    ElementWise {
        /// Input tensor
        input: Vec<f32>,
        /// Result tensor
        output: Arc<parking_lot::Mutex<Vec<f32>>>,
        /// Operation to perform
        op: ElementWiseOp,
    },
    /// Convolution operation for neural networks
    Convolution {
        /// Input tensor (batch_size, height, width, channels)
        input: Vec<f32>,
        /// Convolution kernel (height, width, in_channels, out_channels)
        kernel: Vec<f32>,
        /// Result tensor
        output: Arc<parking_lot::Mutex<Vec<f32>>>,
        /// Input dimensions (batch_size, height, width, channels)
        input_dims: (usize, usize, usize, usize),
        /// Kernel dimensions (height, width, in_channels, out_channels)
        kernel_dims: (usize, usize, usize, usize),
        /// Stride (height, width)
        stride: (usize, usize),
        /// Padding (height, width)
        padding: (usize, usize),
    },
}

/// Element-wise operations for tensor manipulation.
///
/// This enumeration defines the element-wise operations that can be
/// performed on tensors, supporting both activation functions for
/// neural networks and basic arithmetic operations.
#[derive(Debug, Clone, Copy)]
pub enum ElementWiseOp {
    /// ReLU activation: max(0, x)
    ReLU,
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh activation: tanh(x)
    Tanh,
    /// Element-wise addition with constant: x + c
    Add(f32),
    /// Element-wise multiplication with constant: x * c
    Multiply(f32),
}

/// Core trait for hardware acceleration operations.
///
/// This trait defines the interface for different hardware acceleration
/// implementations, enabling polymorphic acceleration across heterogeneous
/// computing architectures. Implementations of this trait provide hardware-specific
/// optimizations while presenting a unified interface to the rest of the system.
pub trait HardwareAccelerator: std::fmt::Debug + Send + Sync {
    /// Returns the type of hardware accelerator.
    fn accelerator_type(&self) -> AcceleratorType;
    
    /// Returns whether this accelerator is available on the current system.
    fn is_available(&self) -> bool;
    
    /// Returns hardware capabilities of this accelerator.
    fn capabilities(&self) -> HardwareCapabilities;
    
    /// Generates lookup tables for faster operations.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to store the generated tables
    ///
    /// # Returns
    ///
    /// Result indicating success or containing an error
    fn generate_lookup_tables(&self, path: &Path) -> Result<()>;
    
    /// Calculate syndromes for error detection.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data bytes
    /// * `syndrome_count` - Number of syndromes to calculate
    ///
    /// # Returns
    ///
    /// Vector of calculated syndromes or an error
    fn calculate_syndromes(&self, data: &[u8], syndrome_count: usize) -> Result<Vec<u16>>;
    
    /// Multiply two arrays of field elements element-wise.
    ///
    /// # Arguments
    ///
    /// * `a` - First input array
    /// * `b` - Second input array
    ///
    /// # Returns
    ///
    /// Result array with element-wise multiplication or an error
    fn multiply_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>>;
    
    /// Add two arrays of field elements element-wise.
    ///
    /// # Arguments
    ///
    /// * `a` - First input array
    /// * `b` - Second input array
    ///
    /// # Returns
    ///
    /// Result array with element-wise addition or an error
    fn add_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>>;
    
    /// Evaluate multiple polynomials at multiple points.
    ///
    /// # Arguments
    ///
    /// * `polys` - Polynomials to evaluate
    /// * `points` - Points at which to evaluate the polynomials
    ///
    /// # Returns
    ///
    /// Matrix of evaluation results or an error
    fn polynomial_eval_batch(&self, polys: &[Vec<u16>], points: &[u16]) -> Result<Vec<Vec<u16>>>;
    
    /// Perform tensor operations for neural-symbolic integration.
    ///
    /// # Arguments
    ///
    /// * `op` - Tensor operation to perform
    ///
    /// # Returns
    ///
    /// Result indicating success or containing an error
    fn perform_tensor_operation(&self, op: TensorOperation) -> Result<()>;
    
    /// Returns the underlying Galois field.
    fn galois_field(&self) -> Arc<GaloisField>;
    
    /// Returns hardware-specific statistics.
    fn get_statistics(&self) -> HardwareStatistics;
    
    /// Checks if the accelerator supports Hamming code operations.
    fn supports_hamming(&self) -> bool {
        false
    }
    
    /// Encodes data using Hamming code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `parity_bits` - Number of parity bits to use
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn hamming_encode(&self, _data: &[u8], _parity_bits: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Hamming encode not supported by this accelerator".into()))
    }
    
    /// Decodes data using Hamming code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `parity_bits` - Number of parity bits used
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn hamming_decode(&self, _data: &[u8], _parity_bits: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Hamming decode not supported by this accelerator".into()))
    }
    
    /// Checks if the accelerator supports BCH code operations.
    fn supports_bch(&self) -> bool {
        false
    }
    
    /// Encodes data using BCH code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `t` - Error correction capability
    /// * `m` - Field size parameter
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn bch_encode(&self, _data: &[u8], _t: usize, _m: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("BCH encode not supported by this accelerator".into()))
    }
    
    /// Decodes data using BCH code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `t` - Error correction capability
    /// * `m` - Field size parameter
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn bch_decode(&self, _data: &[u8], _t: usize, _m: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("BCH decode not supported by this accelerator".into()))
    }
    
    /// Checks if the accelerator supports Reed-Muller code operations.
    fn supports_reed_muller(&self) -> bool {
        false
    }
    
    /// Encodes data using Reed-Muller code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `r` - Order parameter
    /// * `m` - Length parameter
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn reed_muller_encode(&self, _data: &[u8], _r: usize, _m: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Reed-Muller encode not supported by this accelerator".into()))
    }
    
    /// Decodes data using Reed-Muller code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `r` - Order parameter
    /// * `m` - Length parameter
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn reed_muller_decode(&self, _data: &[u8], _r: usize, _m: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Reed-Muller decode not supported by this accelerator".into()))
    }
    
    /// Checks if the accelerator supports Turbo code operations.
    fn supports_turbo(&self) -> bool {
        false
    }
    
    /// Encodes data using Turbo code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `constraint_length` - Constraint length parameter
    /// * `code_rate` - Code rate parameter
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn turbo_encode(&self, _data: &[u8], _constraint_length: usize, _code_rate: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Turbo encode not supported by this accelerator".into()))
    }
    
    /// Decodes data using Turbo code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `constraint_length` - Constraint length parameter
    /// * `code_rate` - Code rate parameter
    /// * `iterations` - Number of iterations for decoding
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn turbo_decode(&self, _data: &[u8], _constraint_length: usize, _code_rate: usize, _iterations: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Turbo decode not supported by this accelerator".into()))
    }
    
    /// Checks if the accelerator supports Convolutional code operations.
    fn supports_convolutional(&self) -> bool {
        false
    }
    
    /// Encodes data using Convolutional code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `constraint_length` - Constraint length parameter
    /// * `generator_polynomials` - Generator polynomials
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn convolutional_encode(&self, _data: &[u8], _constraint_length: usize, _generator_polynomials: &[u64]) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Convolutional encode not supported by this accelerator".into()))
    }
    
    /// Decodes data using Convolutional code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `constraint_length` - Constraint length parameter
    /// * `generator_polynomials` - Generator polynomials
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn convolutional_decode(&self, _data: &[u8], _constraint_length: usize, _generator_polynomials: &[u64]) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Convolutional decode not supported by this accelerator".into()))
    }
    
    /// Checks if the accelerator supports Fountain code operations.
    fn supports_fountain(&self) -> bool {
        false
    }
    
    /// Encodes data using Fountain code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `source_symbols` - Number of source symbols
    /// * `symbol_size` - Size of each symbol in bytes
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn fountain_encode(&self, _data: &[u8], _source_symbols: usize, _symbol_size: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Fountain encode not supported by this accelerator".into()))
    }
    
    /// Decodes data using Fountain code with hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `source_symbols` - Number of source symbols
    /// * `symbol_size` - Size of each symbol in bytes
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn fountain_decode(&self, _data: &[u8], _source_symbols: usize, _symbol_size: usize) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("Fountain decode not supported by this accelerator".into()))
    }
    
    /// Returns whether the accelerator supports XypherGrid operations.
    fn supports_xypher_grid(&self) -> bool {
        false
    }
    
    /// Initialize XypherGrid tables using this accelerator
    ///
    /// # Arguments
    ///
    /// * `xypher_grid` - XypherGrid instance
    ///
    /// # Returns
    ///
    /// Result indicating success or containing an error
    fn initialize_xypher_grid(&self, _xypher_grid: &XypherGrid) -> Result<()> {
        Err(Error::UnsupportedOperation("XypherGrid initialization not supported by this accelerator".into()))
    }
    
    /// Use XypherGrid tables to accelerate encoding operations
    ///
    /// # Arguments
    ///
    /// * `data` - Input data to encode
    /// * `algorithm` - Encoding algorithm
    /// * `params` - Algorithm parameters
    ///
    /// # Returns
    ///
    /// Encoded data or an error
    fn xypher_grid_encode(&self, _data: &[u8], _algorithm: &str, _params: &[usize]) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("XypherGrid encoding not supported by this accelerator".into()))
    }
    
    /// Use XypherGrid tables to accelerate decoding operations
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded data to decode
    /// * `algorithm` - Decoding algorithm
    /// * `params` - Algorithm parameters
    ///
    /// # Returns
    ///
    /// Decoded data or an error
    fn xypher_grid_decode(&self, _data: &[u8], _algorithm: &str, _params: &[usize]) -> Result<Vec<u8>> {
        Err(Error::UnsupportedOperation("XypherGrid decoding not supported by this accelerator".into()))
    }
}

/// Composite accelerator that intelligently distributes work across multiple
/// hardware acceleration devices.
///
/// This implementation orchestrates workload distribution across heterogeneous
/// computing resources, selecting the optimal accelerator for each operation
/// based on the operation characteristics and current device load.
#[derive(Debug)]
pub struct CompositeAccelerator {
    /// List of hardware accelerators available for workload distribution
    accelerators: Vec<Arc<dyn HardwareAccelerator>>,
    /// Galois field for arithmetic operations
    galois_field: Arc<GaloisField>,
    /// Statistics for hardware operations
    stats: RwLock<HardwareStatistics>,
    /// Cache for accelerator selection decisions
    decision_cache: RwLock<HashMap<String, usize>>,
}

impl CompositeAccelerator {
    /// Creates a new composite accelerator.
    ///
    /// # Arguments
    ///
    /// * `accelerators` - List of hardware accelerators
    ///
    /// # Returns
    ///
    /// A new composite accelerator
    ///
    /// # Panics
    ///
    /// Panics if no accelerators are provided
    pub fn new(accelerators: Vec<Arc<dyn HardwareAccelerator>>) -> Self {
        if accelerators.is_empty() {
            panic!("Composite accelerator requires at least one hardware accelerator");
        }
        
        // Use the first accelerator's Galois field
        let galois_field = accelerators[0].galois_field();
        
        Self {
            accelerators,
            galois_field,
            stats: RwLock::new(HardwareStatistics::default()),
            decision_cache: RwLock::new(HashMap::new()),
        }
    }
    
    /// Selects the optimal accelerator for a given operation based on heuristics.
    ///
    /// This method employs a sophisticated decision-making algorithm to select
    /// the most appropriate hardware accelerator for a specific operation,
    /// considering factors such as:
    ///
    /// - Operation characteristics (compute vs. memory bound)
    /// - Data size and structure
    /// - Available hardware capabilities
    /// - Current device load and utilization
    /// - Historical performance for similar operations
    ///
    /// # Arguments
    ///
    /// * `operation_type` - Type of operation to be performed
    /// * `data_size` - Size of data to be processed (in elements)
    /// * `complexity` - Computational complexity factor (higher = more compute intensive)
    ///
    /// # Returns
    ///
    /// Reference to the selected hardware accelerator
    fn select_optimal_accelerator(
        &self,
        operation_type: &str,
        data_size: usize,
        complexity: f32,
    ) -> Arc<dyn HardwareAccelerator> {
        // Check decision cache first
        let cache_key = format!("{}:{}:{}", operation_type, data_size, complexity);
        {
            let cache = self.decision_cache.read().unwrap();
            if let Some(&idx) = cache.get(&cache_key) {
                if idx < self.accelerators.len() {
                    return Arc::clone(&self.accelerators[idx]);
                }
            }
        }
        
        // Categorize operation characteristics
        let compute_bound = complexity > 1.5;
        let memory_bound = data_size > 1_000_000;
        let parallel_friendly = data_size > 1000;
        
        // Score each accelerator based on operation characteristics
        let mut scores = Vec::with_capacity(self.accelerators.len());
        
        for accelerator in &self.accelerators {
            let caps = accelerator.capabilities();
            let acc_type = accelerator.accelerator_type();
            
            // Base score starts at 1.0
            let mut score = 1.0;
            
            // Consider accelerator type
            match acc_type {
                // CUDA excels at compute-bound, parallel-friendly operations
                AcceleratorType::Cuda if compute_bound && parallel_friendly => {
                    score *= 2.0;
                    if data_size > 10_000_000 {
                        score *= 1.5; // CUDA especially good for large datasets
                    }
                }
                
                // OpenCL is versatile but particularly good for medium workloads
                AcceleratorType::OpenCL if parallel_friendly => {
                    score *= 1.5;
                    if !compute_bound && memory_bound {
                        score *= 1.2; // OpenCL handles memory-bound ops well
                    }
                }
                
                // CPU with AVX2 is good for smaller workloads and specific operations
                AcceleratorType::Cpu if caps.avx2_available => {
                    if !memory_bound && compute_bound {
                        score *= 1.3; // AVX2 good for compute-bound, cache-friendly ops
                    }
                    if data_size < 10_000 {
                        score *= 1.5; // CPU often better for small datasets (less overhead)
                    }
                }
                
                // Default adjustments based on accelerator type
                AcceleratorType::Cuda => score *= 1.3,
                AcceleratorType::OpenCL => score *= 1.1,
                AcceleratorType::Cpu => score *= 1.0,
                AcceleratorType::Composite => score *= 0.5, // Avoid recursive composites
            }
            
            // Consider available memory
            if memory_bound && caps.available_memory < data_size * 4 {
                score *= 0.5; // Penalize if memory might be insufficient
            }
            
            // Consider operation specific capabilities
            if operation_type.contains("galois") && acc_type == AcceleratorType::Cuda {
                score *= 1.2; // CUDA optimized for Galois field operations
            }
            if operation_type.contains("matrix") && caps.avx2_available {
                score *= 1.1; // AVX2 good for matrix operations
            }
            if operation_type.contains("neural") && acc_type == AcceleratorType::Cuda {
                score *= 1.3; // CUDA excellent for neural network ops
            }
            
            scores.push((score, accelerator));
        }
        
        // Select accelerator with highest score
        scores.sort_by(|(a_score, _), (b_score, _)| b_score.partial_cmp(a_score).unwrap());
        let best_idx = self.accelerators.iter().position(|a| 
            Arc::ptr_eq(a, &scores[0].1)
        ).unwrap_or(0);
        
        // Cache the decision
        {
            let mut cache = self.decision_cache.write().unwrap();
            cache.insert(cache_key, best_idx);
            
            // Prune cache if it grows too large
            if cache.len() > 100 {
                // Remove random entries to avoid pattern-based performance degradation
                let keys: Vec<String> = cache.keys().take(20).cloned().collect();
                for key in keys {
                    cache.remove(&key);
                }
            }
        }
        
        Arc::clone(&self.accelerators[best_idx])
    }
}

impl HardwareAccelerator for CompositeAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::Composite
    }

    fn is_available(&self) -> bool {
        // Composite accelerator is available if at least one of its accelerators is available
        self.accelerators.iter().any(|acc| acc.is_available())
    }

    fn capabilities(&self) -> HardwareCapabilities {
        let mut caps = HardwareCapabilities {
            avx2_available: false,
            openmp_available: false,
            cuda_available: false,
            cuda_device_count: 0,
            opencl_available: false,
            opencl_device_count: 0,
            processor_count: 0,
            available_memory: 0,
            hamming_available: false,
            bch_available: false,
            reed_muller_available: false,
            turbo_available: false,
            convolutional_available: false,
            fountain_available: false,
        };
        
        // Combine capabilities from all accelerators
        for acc in &self.accelerators {
            let acc_caps = acc.capabilities();
            caps.avx2_available |= acc_caps.avx2_available;
            caps.openmp_available |= acc_caps.openmp_available;
            caps.cuda_available |= acc_caps.cuda_available;
            caps.cuda_device_count += acc_caps.cuda_device_count;
            caps.opencl_available |= acc_caps.opencl_available;
            caps.opencl_device_count += acc_caps.opencl_device_count;
            caps.processor_count += acc_caps.processor_count;
            caps.available_memory += acc_caps.available_memory;
            caps.hamming_available |= acc_caps.hamming_available;
            caps.bch_available |= acc_caps.bch_available;
            caps.reed_muller_available |= acc_caps.reed_muller_available;
            caps.turbo_available |= acc_caps.turbo_available;
            caps.convolutional_available |= acc_caps.convolutional_available;
            caps.fountain_available |= acc_caps.fountain_available;
        }
        
        caps
    }

    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the composite-specific tables directory
        let composite_path = path.join("composite");
        std::fs::create_dir_all(&composite_path)?;
        
        // Generate lookup tables for all accelerators
        for (i, acc) in self.accelerators.iter().enumerate() {
            let acc_path = composite_path.join(format!("acc_{}", i));
            if let Err(e) = acc.generate_lookup_tables(&acc_path) {
                // Log error but continue with other accelerators
                tracing::warn!("Failed to generate lookup tables for accelerator {}: {}", i, e);
            }
        }
        
        // Generate Galois field lookup tables
        self.galois_field().generate_lookup_tables()?;
        
        Ok(())
    }

    fn calculate_syndromes(&self, data: &[u8], syndrome_count: usize) -> Result<Vec<u16>> {
        // Select optimal accelerator for syndrome calculation
        let accelerator = self.select_optimal_accelerator(
            "galois_syndromes",
            data.len() + syndrome_count,
            1.2
        );
        
        // Delegate operation
        let result = accelerator.calculate_syndromes(data, syndrome_count);
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.sync_operations += 1;
            
            // Update accelerator-specific stats based on the type used
            match accelerator.accelerator_type() {
                AcceleratorType::Cpu => stats.cpu_operations += 1,
                AcceleratorType::Cuda => stats.cuda_operations += 1,
                AcceleratorType::OpenCL => stats.opencl_operations += 1,
                _ => {}
            }
        }
        
        result
    }

    fn multiply_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput("Input arrays must have the same length".into()));
        }
        
        // Select optimal accelerator for Galois field multiplication
        let accelerator = self.select_optimal_accelerator(
            "galois_multiply",
            a.len(),
            1.0
        );
        
        // Delegate operation
        let result = accelerator.multiply_vec(a, b);
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.sync_operations += 1;
            
            // Update accelerator-specific stats based on the type used
            match accelerator.accelerator_type() {
                AcceleratorType::Cpu => stats.cpu_operations += 1,
                AcceleratorType::Cuda => stats.cuda_operations += 1,
                AcceleratorType::OpenCL => stats.opencl_operations += 1,
                _ => {}
            }
        }
        
        result
    }

    fn add_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput("Input arrays must have the same length".into()));
        }
        
        // Select optimal accelerator for Galois field addition
        let accelerator = self.select_optimal_accelerator(
            "galois_add",
            a.len(),
            0.8 // Addition is less compute intensive than multiplication
        );
        
        // Delegate operation
        let result = accelerator.add_vec(a, b);
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.sync_operations += 1;
            
            // Update accelerator-specific stats based on the type used
            match accelerator.accelerator_type() {
                AcceleratorType::Cpu => stats.cpu_operations += 1,
                AcceleratorType::Cuda => stats.cuda_operations += 1,
                AcceleratorType::OpenCL => stats.opencl_operations += 1,
                _ => {}
            }
        }
        
        result
    }

    fn polynomial_eval_batch(&self, polys: &[Vec<u16>], points: &[u16]) -> Result<Vec<Vec<u16>>> {
        // Calculate total data size
        let data_size = polys.iter().map(|p| p.len()).sum::<usize>() + points.len();
        
        // Polynomial evaluation is computationally intensive
        let complexity = 1.8;
        
        // Select optimal accelerator for polynomial evaluation
        let accelerator = self.select_optimal_accelerator(
            "polynomial_eval",
            data_size,
            complexity
        );
        
        // Delegate operation
        let result = accelerator.polynomial_eval_batch(polys, points);
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.sync_operations += 1;
            
            // Update accelerator-specific stats based on the type used
            match accelerator.accelerator_type() {
                AcceleratorType::Cpu => stats.cpu_operations += 1,
                AcceleratorType::Cuda => stats.cuda_operations += 1,
                AcceleratorType::OpenCL => stats.opencl_operations += 1,
                _ => {}
            }
        }
        
        result
    }

    fn perform_tensor_operation(&self, op: TensorOperation) -> Result<()> {
        // Calculate operation size and complexity
        let (data_size, operation_type, complexity) = match &op {
            TensorOperation::MatrixMultiply { a, b, c: _, dims } => {
                let (m, _k, n) = *dims;
                
                // Matrix multiplication is O(m*n*k) complexity
                let size = a.len() + b.len() + (m * n);
                let complexity = 2.0; // Matrix multiplication is compute-intensive
                
                (size, "matrix_multiply", complexity)
            }
            TensorOperation::ElementWise { input, output: _, op: _ } => {
                let size = input.len() * 2;
                let complexity = 0.9; // Element-wise operations are relatively simple
                
                (size, "element_wise", complexity)
            }
            TensorOperation::Convolution { input, kernel, output: _, input_dims, kernel_dims, stride: _, padding: _ } => {
                let (batch_size, input_height, input_width, _input_channels) = *input_dims;
                let (_kernel_height, _kernel_width, _, output_channels) = *kernel_dims;
                
                // Convolution is highly compute-intensive
                let size = input.len() + kernel.len() + 
                           (batch_size * input_height * input_width * output_channels);
                let complexity = 2.2; // Convolutions are very compute-intensive
                
                (size, "convolution", complexity)
            }
        };
        
        // Select optimal accelerator
        let accelerator = self.select_optimal_accelerator(
            operation_type,
            data_size,
            complexity
        );
        
        // Delegate operation
        let result = accelerator.perform_tensor_operation(op);
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.sync_operations += 1;
            
            // Update accelerator-specific stats based on the type used
            match accelerator.accelerator_type() {
                AcceleratorType::Cpu => stats.cpu_operations += 1,
                AcceleratorType::Cuda => stats.cuda_operations += 1,
                AcceleratorType::OpenCL => stats.opencl_operations += 1,
                _ => {}
            }
        }
        
        result
    }

    fn galois_field(&self) -> Arc<GaloisField> {
        self.galois_field.clone()
    }

    fn get_statistics(&self) -> HardwareStatistics {
        // Combine base statistics with those from all accelerators
        let mut stats = self.stats.read().unwrap().clone();
        
        for acc in &self.accelerators {
            let acc_stats = acc.get_statistics();
            stats.sync_operations += acc_stats.sync_operations;
            stats.async_operations += acc_stats.async_operations;
            stats.cpu_operations += acc_stats.cpu_operations;
            stats.cuda_operations += acc_stats.cuda_operations;
            stats.opencl_operations += acc_stats.opencl_operations;
            stats.avx2_operations += acc_stats.avx2_operations;
            stats.total_time_ms += acc_stats.total_time_ms;
        }
        
        stats
    }
    
    // Override methods for specific error correction codes by delegating to a suitable accelerator
    
    fn supports_hamming(&self) -> bool {
        self.accelerators.iter().any(|acc| acc.supports_hamming())
    }
    
    fn hamming_encode(&self, data: &[u8], parity_bits: usize) -> Result<Vec<u8>> {
        // Find an accelerator that supports Hamming encoding
        for acc in &self.accelerators {
            if acc.supports_hamming() {
                return acc.hamming_encode(data, parity_bits);
            }
        }
        
        Err(Error::UnsupportedOperation("No accelerator supports Hamming encode".into()))
    }
    
    fn hamming_decode(&self, data: &[u8], parity_bits: usize) -> Result<Vec<u8>> {
        // Find an accelerator that supports Hamming decoding
        for acc in &self.accelerators {
            if acc.supports_hamming() {
                return acc.hamming_decode(data, parity_bits);
            }
        }
        
        Err(Error::UnsupportedOperation("No accelerator supports Hamming decode".into()))
    }
    
    fn supports_bch(&self) -> bool {
        self.accelerators.iter().any(|acc| acc.supports_bch())
    }
    
    fn bch_encode(&self, data: &[u8], t: usize, m: usize) -> Result<Vec<u8>> {
        // Find an accelerator that supports BCH encoding
        for acc in &self.accelerators {
            if acc.supports_bch() {
                return acc.bch_encode(data, t, m);
            }
        }
        
        Err(Error::UnsupportedOperation("No accelerator supports BCH encode".into()))
    }
    
    fn bch_decode(&self, data: &[u8], t: usize, m: usize) -> Result<Vec<u8>> {
        // Find an accelerator that supports BCH decoding
        for acc in &self.accelerators {
            if acc.supports_bch() {
                return acc.bch_decode(data, t, m);
            }
        }
        
        Err(Error::UnsupportedOperation("No accelerator supports BCH decode".into()))
    }
    
    fn supports_xypher_grid(&self) -> bool {
        self.accelerators.iter().any(|acc| acc.supports_xypher_grid())
    }
    
    fn initialize_xypher_grid(&self, xypher_grid: &XypherGrid) -> Result<()> {
        // Find an accelerator that supports XypherGrid initialization
        for acc in &self.accelerators {
            if acc.supports_xypher_grid() {
                return acc.initialize_xypher_grid(xypher_grid);
            }
        }
        
        Err(Error::UnsupportedOperation("No accelerator supports XypherGrid initialization".into()))
    }
}

/// Hardware acceleration manager to provide a unified interface for all
/// acceleration operations.
///
/// This manager initializes and maintains all available hardware accelerators,
/// enabling centralized access, configuration, and monitoring of acceleration
/// resources. It serves as the primary entry point for client code to access
/// hardware acceleration capabilities.
#[derive(Debug)]
pub struct HardwareAccelerationManager {
    /// List of available hardware accelerators
    accelerators: Vec<Arc<dyn HardwareAccelerator>>,
    /// Composite accelerator for intelligent workload distribution
    composite_accelerator: Arc<CompositeAccelerator>,
    /// Path to lookup tables
    tables_path: std::path::PathBuf,
}

impl HardwareAccelerationManager {
    /// Creates a new hardware acceleration manager.
    ///
    /// # Arguments
    ///
    /// * `tables_path` - Path to store lookup tables
    ///
    /// # Returns
    ///
    /// A new hardware acceleration manager or an error
    pub fn new(tables_path: impl AsRef<Path>) -> Result<Self> {
        // Create tables path directory if it doesn't exist
        let tables_path = tables_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&tables_path)?;
        
        // Initialize hardware accelerators
        let mut accelerators = Vec::new();
        
        // Register CPU accelerators
        if let Err(e) = register_cpu_accelerators(&mut accelerators) {
            tracing::warn!("Failed to register CPU accelerators: {}", e);
        }
        
        // Register CUDA accelerators
        if let Err(e) = register_cuda_accelerators(&mut accelerators) {
            tracing::warn!("Failed to register CUDA accelerators: {}", e);
        }
        
        // Register OpenCL accelerators
        if let Err(e) = register_opencl_accelerators(&mut accelerators) {
            tracing::warn!("Failed to register OpenCL accelerators: {}", e);
        }
        
        // Ensure we have at least one accelerator
        if accelerators.is_empty() {
            return Err(Error::HardwareUnavailable("No hardware accelerators available".into()));
        }
        
        // Create composite accelerator
        let composite_accelerator = Arc::new(CompositeAccelerator::new(accelerators.clone()));
        
        // Register composite accelerator
        accelerators.push(composite_accelerator.clone());
        
        Ok(Self {
            accelerators,
            composite_accelerator,
            tables_path,
        })
    }
    
    /// Returns a reference to the composite accelerator.
    ///
    /// The composite accelerator provides intelligent workload distribution
    /// across all available hardware acceleration devices.
    ///
    /// # Returns
    ///
    /// Reference to the composite accelerator
    pub fn accelerator(&self) -> Arc<dyn HardwareAccelerator> {
        self.composite_accelerator.clone()
    }
    
    /// Returns a reference to a specific accelerator by type.
    ///
    /// # Arguments
    ///
    /// * `accelerator_type` - Type of accelerator to get
    ///
    /// # Returns
    ///
    /// Reference to the requested accelerator or None if not available
    pub fn get_accelerator(&self, accelerator_type: AcceleratorType) -> Option<Arc<dyn HardwareAccelerator>> {
        self.accelerators.iter()
            .find(|acc| acc.accelerator_type() == accelerator_type)
            .cloned()
    }
    
    /// Returns a list of all available accelerators.
    ///
    /// # Returns
    ///
    /// List of all available accelerators
    pub fn list_accelerators(&self) -> Vec<Arc<dyn HardwareAccelerator>> {
        self.accelerators.clone()
    }
    
    /// Generates lookup tables for all accelerators.
    ///
    /// # Returns
    ///
    /// Result indicating success or containing an error
    pub fn generate_lookup_tables(&self) -> Result<()> {
        for acc in &self.accelerators {
            if let Err(e) = acc.generate_lookup_tables(&self.tables_path) {
                tracing::warn!("Failed to generate lookup tables for accelerator {:?}: {}", 
                              acc.accelerator_type(), e);
            }
        }
        Ok(())
    }
    
    /// Returns statistics for all accelerators.
    ///
    /// # Returns
    ///
    /// Map of accelerator type to statistics
    pub fn get_statistics(&self) -> HashMap<AcceleratorType, HardwareStatistics> {
        self.accelerators.iter()
            .map(|acc| (acc.accelerator_type(), acc.get_statistics()))
            .collect()
    }
}

/// Initializes the hardware acceleration system.
///
/// This function is the entry point for client code to initialize and access
/// the hardware acceleration infrastructure. It discovers available hardware,
/// initializes appropriate accelerators, and returns a manager that provides
/// a unified interface to all acceleration capabilities.
///
/// # Arguments
///
/// * `tables_path` - Path to store lookup tables
///
/// # Returns
///
/// A hardware acceleration manager or an error
pub fn initialize_hardware_acceleration(tables_path: impl AsRef<Path>) -> Result<HardwareAccelerationManager> {
    HardwareAccelerationManager::new(tables_path)
}