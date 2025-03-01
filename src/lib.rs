//! # HyVERX
//!
//! HyVERX is an advanced multi-dimensional error correction system with hardware acceleration.
//! It provides high-performance error detection and correction through various algorithms,
//! hardware-accelerated operations, and neural-symbolic integration.
//!
//! ## Features
//!
//! - Multi-dimensional error detection and correction with parallel processing
//! - Dynamic algorithm allocation based on error characteristics
//! - Hardware acceleration via AVX2/OpenMP SIMT and CUDA/cuDNN tensor cores
//! - Neural-symbolic integration for complex error pattern recognition
//! - Precomputed correction matrices for zero-cost abstraction
//!
//! ## Modules
//!
//! - `config`: Configuration settings for the HyVERX system
//! - `galois`: Galois field operations for finite field arithmetic
//! - `hardware`: Hardware acceleration implementations (CPU, CUDA, OpenCL)
//! - `neural`: Neural-symbolic integration for error pattern learning
//! - `algorithms`: Error correction algorithms (Reed-Solomon, etc.)
//! - `parallel`: Parallel processing utilities for multi-dimensional correction
//! - `utils`: Utility functions and common data structures
//! - `ffi`: Foreign Function Interface for interoperability
//! - `python`: Python bindings using PyO3
//! - `xypher_grid`: Module for XYpher grid operations

#![allow(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]

use std::sync::Arc;

// Re-export error types
pub use crate::error::{Error, Result};

// Modules
pub mod algorithms;
pub mod config;
pub mod error;
pub mod galois;
pub mod hardware;
pub mod neural;
pub mod parallel;
pub mod xypher_grid;
pub mod prelude {
    //! Prelude module that re-exports commonly used types and functions.

    pub use crate::algorithms::{
        AdaptiveReedSolomon, ConvolutionalCode, ErrorCorrectionAlgorithm, Ldpc, ReedSolomon,
        TensorReedSolomon, TurboCode,
    };
    pub use crate::config::{Config, HardwareTarget};
    pub use crate::error::{Error, Result};
    pub use crate::galois::GaloisField;
    pub use crate::hardware::cpu::CPUAccelerator;
    pub use crate::hardware::dgpu::GPUAccelerator;
    pub use crate::hardware::igpu::IGPUAccelerator;
    pub use crate::hardware::AcceleratorType;
    pub use crate::hardware::{HardwareAccelerator, HardwareCapabilities, TensorOperation};
    pub use crate::neural::{
        ErrorAnalyzer, ErrorPattern, NeuralGaloisCorrector, PatternRecognizer,
    };
    pub use crate::parallel::{AlgorithmAllocator, DimensionMapper, ThreadPool, WorkScheduler};
    pub use crate::xypher_grid::{get_xypher_grid, initialize_tables, XypherGrid};
    pub use crate::HyVerxSystem;
}

/// Statistics and performance metrics for error correction operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Statistics {
    /// Number of encoding operations performed
    pub encode_count: usize,
    /// Number of decoding operations performed
    pub decode_count: usize,
    /// Number of errors detected
    pub errors_detected: usize,
    /// Number of errors corrected
    pub errors_corrected: usize,
    /// Error correction success rate (0.0-1.0)
    pub success_rate: f64,
    /// Error pattern learning statistics
    pub pattern_learning: PatternLearningStats,
    /// Hardware acceleration usage statistics
    pub hardware_usage: HardwareUsageStats,
    /// Algorithm allocation statistics
    pub algorithm_allocation: AlgorithmAllocationStats,
}

/// Statistics for error pattern learning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternLearningStats {
    /// Number of patterns learned
    pub patterns_learned: usize,
    /// Number of successful pattern matches
    pub pattern_hits: usize,
    /// Number of pattern match attempts that failed
    pub pattern_misses: usize,
    /// Pattern match success rate (0.0-1.0)
    pub hit_rate: f64,
}

/// Statistics for hardware acceleration usage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HardwareUsageStats {
    /// CPU operations count
    pub cpu_operations: usize,
    /// CUDA operations count
    pub cuda_operations: usize,
    /// OpenCL operations count
    pub opencl_operations: usize,
    /// AVX2 operations count
    pub avx2_operations: usize,
    /// Total processing time (ms)
    pub total_time_ms: f64,
}

/// Statistics for algorithm allocation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlgorithmAllocationStats {
    /// Reed-Solomon usage count
    pub reed_solomon_count: usize,
    /// LDPC usage count
    pub ldpc_count: usize,
    /// Turbo code usage count
    pub turbo_count: usize,
    /// Convolutional code usage count
    pub convolutional_count: usize,
    /// Tensor Reed-Solomon usage count
    pub tensor_rs_count: usize,
    /// Adaptive Reed-Solomon usage count
    pub adaptive_rs_count: usize,
    /// Parallel Reed-Solomon usage count
    pub parallel_rs_count: usize,
}

/// Main entry point for the HyVERX system.
#[derive(Debug)]
pub struct HyVerxSystem {
    /// System configuration
    config: config::Config,
    /// Galois field for finite field arithmetic
    galois_field: Arc<galois::GaloisField>,
    /// Hardware accelerator for hardware-specific optimizations
    hardware_accelerator: Arc<dyn hardware::HardwareAccelerator>,
    /// Algorithm allocator for dynamic algorithm selection
    algorithm_allocator: parallel::AlgorithmAllocator,
    /// Error analyzer for pattern recognition
    error_analyzer: neural::ErrorAnalyzer,
    /// XypherGrid for precomputed tables
    #[allow(dead_code)]
    xypher_grid: Arc<xypher_grid::XypherGrid>,
    /// Statistics collector
    stats: Statistics,
}

impl HyVerxSystem {
    /// Creates a new HyVERX system with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration settings for the system
    ///
    /// # Returns
    ///
    /// A new HyVERX system instance
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails, such as hardware detection issues
    pub fn new(config: config::Config) -> Result<Self> {
        // Initialize Galois field
        let galois_field = Arc::new(galois::GaloisField::new(config.field_polynomial()));

        // Initialize hardware accelerator
        let hardware_accelerator =
            Self::create_hardware_accelerator(&config, Arc::clone(&galois_field))?;

        // Initialize XypherGrid
        let xypher_grid = Arc::new(xypher_grid::XypherGrid::new());
        xypher_grid.initialize_precomputed_tables()?;

        // Create thread pool and work scheduler
        let thread_pool = Arc::new(parallel::ThreadPool::new(parallel::ParallelConfig {
            thread_count: config.max_threads(),
            ..parallel::ParallelConfig::default()
        }));
        let work_scheduler = Arc::new(parallel::WorkScheduler::new(
            thread_pool,
            parallel::ParallelConfig::default(),
        ));

        // Initialize algorithm allocator
        let mut algorithm_allocator = parallel::AlgorithmAllocator::new(work_scheduler);
        algorithm_allocator.with_algorithms(vec![
            algorithms::AlgorithmType::ReedSolomon,
            algorithms::AlgorithmType::Ldpc,
            algorithms::AlgorithmType::PolarCode,
            algorithms::AlgorithmType::HammingCode,
            algorithms::AlgorithmType::BchCode,
            algorithms::AlgorithmType::ReedMullerCode,
            algorithms::AlgorithmType::Turbo,
            algorithms::AlgorithmType::Convolutional,
            algorithms::AlgorithmType::Fountain,
        ]);

        // Initialize error analyzer
        let error_analyzer = neural::ErrorAnalyzer::new(
            config.max_data_size(),
            config.dimensions(),
            Arc::clone(&hardware_accelerator),
        );

        // Initialize statistics
        let stats = Statistics {
            encode_count: 0,
            decode_count: 0,
            errors_detected: 0,
            errors_corrected: 0,
            success_rate: 0.0,
            pattern_learning: PatternLearningStats {
                patterns_learned: 0,
                pattern_hits: 0,
                pattern_misses: 0,
                hit_rate: 0.0,
            },
            hardware_usage: HardwareUsageStats {
                cpu_operations: 0,
                cuda_operations: 0,
                opencl_operations: 0,
                avx2_operations: 0,
                total_time_ms: 0.0,
            },
            algorithm_allocation: AlgorithmAllocationStats {
                reed_solomon_count: 0,
                ldpc_count: 0,
                turbo_count: 0,
                convolutional_count: 0,
                tensor_rs_count: 0,
                adaptive_rs_count: 0,
                parallel_rs_count: 0,
            },
        };

        Ok(Self {
            config,
            galois_field,
            hardware_accelerator,
            algorithm_allocator,
            error_analyzer,
            xypher_grid,
            stats,
        })
    }

    /// Creates a hardware accelerator based on the configuration.
    fn create_hardware_accelerator(
        config: &config::Config,
        galois_field: Arc<galois::GaloisField>,
    ) -> Result<Arc<dyn hardware::HardwareAccelerator>> {
        // Select hardware accelerator based on configuration and available hardware
        match config.hardware_target() {
            config::HardwareTarget::Auto => {
                // Automatically detect and use the best available hardware
                if hardware::dgpu::GPUAccelerator::is_available() {
                    Ok(Arc::new(hardware::dgpu::GPUAccelerator::new(galois_field)?))
                } else if hardware::igpu::IGPUAccelerator::is_available() {
                    Ok(Arc::new(hardware::igpu::IGPUAccelerator::new(
                        galois_field,
                    )?))
                } else {
                    Ok(Arc::new(hardware::cpu::CPUAccelerator::new(galois_field)?))
                }
            }
            config::HardwareTarget::Cpu => {
                Ok(Arc::new(hardware::cpu::CPUAccelerator::new(galois_field)?))
            }
            config::HardwareTarget::Cuda => {
                if hardware::dgpu::GPUAccelerator::is_available() {
                    Ok(Arc::new(hardware::dgpu::GPUAccelerator::new(galois_field)?))
                } else {
                    Err(Error::HardwareUnavailable(
                        "CUDA is not available on this system".into(),
                    ))
                }
            }
            config::HardwareTarget::OpenCL => {
                if hardware::igpu::IGPUAccelerator::is_available() {
                    Ok(Arc::new(hardware::igpu::IGPUAccelerator::new(
                        galois_field,
                    )?))
                } else {
                    Err(Error::HardwareUnavailable(
                        "OpenCL is not available on this system".into(),
                    ))
                }
            }
            config::HardwareTarget::All => {
                // Use a composite accelerator that combines all available hardware
                let mut accelerators: Vec<Arc<dyn hardware::HardwareAccelerator>> = Vec::new();

                // Always add CPU accelerator
                accelerators.push(Arc::new(hardware::cpu::CPUAccelerator::new(
                    galois_field.clone(),
                )?));

                // Add CUDA accelerator if available
                if hardware::dgpu::GPUAccelerator::is_available() {
                    accelerators.push(Arc::new(hardware::dgpu::GPUAccelerator::new(
                        galois_field.clone(),
                    )?));
                }

                // Add OpenCL accelerator if available
                if hardware::igpu::IGPUAccelerator::is_available() {
                    accelerators.push(Arc::new(hardware::igpu::IGPUAccelerator::new(
                        galois_field.clone(),
                    )?));
                }

                Ok(Arc::new(hardware::CompositeAccelerator::new(accelerators)))
            }
        }
    }

    /// Generates precomputed lookup tables for faster operations.
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or an error if table generation fails
    pub fn generate_lookup_tables(&mut self) -> Result<()> {
        // Generate Galois field lookup tables
        self.galois_field.generate_lookup_tables()?;

        // Create algorithm-specific lookup tables
        let table_path = self.config.table_path();
        if !table_path.exists() {
            std::fs::create_dir_all(table_path)?;
        }

        // Generate Reed-Solomon lookup tables
        let rs_tables_path = table_path.join("reed_solomon");
        if !rs_tables_path.exists() {
            std::fs::create_dir_all(&rs_tables_path)?;
        }

        // Generate polynomial tables
        let poly_path = rs_tables_path.join("polynomials");
        let algorithm = algorithms::create_algorithm(
            algorithms::AlgorithmType::ReedSolomon,
            self.galois_field.clone(),
            self.hardware_accelerator.clone(),
        )?;
        algorithm.generate_lookup_tables(&poly_path)?;

        // Generate syndrome tables
        let syndrome_path = rs_tables_path.join("syndromes");
        algorithm.generate_lookup_tables(&syndrome_path)?;

        // Generate hardware-specific lookup tables
        self.hardware_accelerator
            .generate_lookup_tables(table_path)?;

        Ok(())
    }

    /// Encodes data using the most appropriate algorithm and hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to encode
    ///
    /// # Returns
    ///
    /// Encoded data with error correction symbols appended
    pub fn encode(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        // Select the most appropriate algorithm based on data characteristics
        let algorithm = self.algorithm_allocator.select_encoding_algorithm(
            data.len(),
            0.01,  // Default error rate
            false, // Assume no burst errors by default
        )?;

        // Track statistics
        self.stats.encode_count += 1;

        // Create the algorithm instance
        let algorithm = algorithms::create_algorithm(
            algorithm,
            self.galois_field.clone(),
            self.hardware_accelerator.clone(),
        )?;

        // Encode the data using the selected algorithm
        let encoded = algorithm.encode(data)?;

        Ok(encoded)
    }

    /// Decodes data, detecting and correcting errors using the most appropriate algorithm.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to decode (with error correction symbols)
    ///
    /// # Returns
    ///
    /// Decoded data with errors corrected (without error correction symbols)
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        // Calculate syndromes to detect errors
        let syndromes = self
            .hardware_accelerator
            .calculate_syndromes(data, self.config.ecc_size())?;

        // Check if there are any errors
        let has_errors = !syndromes.iter().all(|&s| s == 0);

        // Track statistics
        self.stats.decode_count += 1;
        if has_errors {
            self.stats.errors_detected += 1;
        }

        if !has_errors {
            // No errors, return the original data without ECC symbols
            let data_len = data.len() - self.config.ecc_size();
            return Ok(data[..data_len].to_vec());
        }

        // Analyze error patterns
        let _error_pattern = self.error_analyzer.analyze_syndromes(
            &syndromes,
            data.len() - self.config.ecc_size(),
            self.config.ecc_size(),
            self.galois_field.field_size(),
        );

        // Select the most appropriate algorithm based on error characteristics
        let algorithm = self.algorithm_allocator.select_decoding_algorithm(
            self.config.ecc_size(),
            0.01,  // Default error rate
            false, // Default to no burst errors
        )?; // Unwrap the Result

        // Create the algorithm instance
        let algorithm_instance = algorithms::create_algorithm(
            algorithm,
            self.galois_field.clone(),
            self.hardware_accelerator.clone(),
        )?;

        // Update statistics based on algorithm selection
        self.update_algorithm_stats(algorithm);

        // Decode the data using the selected algorithm
        let start_time = std::time::Instant::now();
        let decoded = algorithm_instance.decode(data)?;
        let elapsed = start_time.elapsed();

        // Update timing statistics
        self.stats.hardware_usage.total_time_ms += elapsed.as_secs_f64() * 1000.0;

        // Check if error correction was successful
        let success = self.verify_correction(&decoded);
        if success {
            self.stats.errors_corrected += 1;
        }

        // Update success rate
        if self.stats.errors_detected > 0 {
            self.stats.success_rate =
                self.stats.errors_corrected as f64 / self.stats.errors_detected as f64;
        }

        Ok(decoded)
    }

    /// Processes data by encoding it with error correction, then simulating decoding.
    ///
    /// This is primarily for testing and benchmarking purposes.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to process
    ///
    /// # Returns
    ///
    /// Processed data
    pub fn process_data(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        // Encode the data
        let encoded = self.encode(data)?;

        // In a real application, the encoded data would be transmitted or stored
        // and potentially corrupted. For this test, we'll decode it directly.
        let decoded = self.decode(&encoded)?;

        Ok(decoded)
    }

    /// Updates algorithm allocation statistics.
    fn update_algorithm_stats(&mut self, algorithm_type: algorithms::AlgorithmType) {
        match algorithm_type {
            algorithms::AlgorithmType::ReedSolomon => {
                self.stats.algorithm_allocation.reed_solomon_count += 1
            }
            algorithms::AlgorithmType::Ldpc => self.stats.algorithm_allocation.ldpc_count += 1,
            algorithms::AlgorithmType::Turbo => self.stats.algorithm_allocation.turbo_count += 1,
            algorithms::AlgorithmType::Convolutional => {
                self.stats.algorithm_allocation.convolutional_count += 1
            }
            algorithms::AlgorithmType::TensorReedSolomon => {
                self.stats.algorithm_allocation.tensor_rs_count += 1
            }
            algorithms::AlgorithmType::AdaptiveReedSolomon => {
                self.stats.algorithm_allocation.adaptive_rs_count += 1
            }
            _ => (),
        }
    }

    /// Verifies the correction of decoded data by re-encoding and checking syndromes.
    fn verify_correction(&self, decoded: &[u8]) -> bool {
        // Re-encode the decoded data
        let rs_result = algorithms::ReedSolomon::new(
            self.galois_field.clone(),
            self.config.ecc_size() + decoded.len(), // codeword_length = message_length + ecc_size
            decoded.len(),                          // message_length
            self.hardware_accelerator.clone(),
        );

        match rs_result {
            Ok(rs) => {
                let algorithm: &dyn algorithms::ErrorCorrectionAlgorithm = &rs;
                match algorithm.encode(decoded) {
                    Ok(reencoded) => {
                        // Calculate syndromes for the re-encoded data
                        match self
                            .hardware_accelerator
                            .calculate_syndromes(&reencoded, self.config.ecc_size())
                        {
                            Ok(syndromes) => {
                                // All syndromes should be zero if correction was successful
                                syndromes.iter().all(|&s| s == 0)
                            }
                            Err(_) => false,
                        }
                    }
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }

    /// Returns the current statistics for the HyVERX system.
    pub fn get_statistics(&self) -> Statistics {
        self.stats.clone()
    }

    /// Returns the Galois field used by the system.
    pub fn galois_field(&self) -> Arc<galois::GaloisField> {
        self.galois_field.clone()
    }

    /// Returns the hardware accelerator used by the system.
    pub fn hardware_accelerator(&self) -> Arc<dyn hardware::HardwareAccelerator> {
        self.hardware_accelerator.clone()
    }

    /// Returns the configuration of the system.
    pub fn config(&self) -> &config::Config {
        &self.config
    }
}
