//! # Error Correction Algorithms
//! 
//! This module provides implementations of various error correction algorithms
//! used by the HyVERX system. These algorithms are designed to detect and correct
//! errors in data transmitted over noisy channels or stored in unreliable media.
//!
//! ## Algorithms, Variants, and Xypher Grids((Precomputed Lookup Tables)Separate Module)
//!
//! - Reed-Solomon: Classic error correction code for burst errors
//! - LDPC (Low-Density Parity-Check): Efficient for large blocks of data
//! - Turbo Codes: High-performance codes using iterative decoding
//! - Convolutional Codes: Stream-oriented codes with good performance
//! - Adaptive Reed-Solomon: Dynamically adjusts parameters based on error patterns
//! - Polar Codes: Capacity-achieving codes with excellent performance at short block lengths
//! - Hamming Codes: Simple codes for single-bit error correction
//! - BCH Codes: Powerful cyclic codes for multiple error correction
//! - Reed-Muller Codes: Linear codes for moderate error correction
//! 
//! ## Variants
//! ### All core algorithms have variants that automatically adapt to:
//! - available resources
//! - error pattern
//! - data size
//! - error rate
//! 
//! ### Adaptive 
//! - Automatically adjusts the error correction code based on the error pattern
//! - Uses a lookup table to store the error correction code
//! - Can handle a wide range of error patterns
//! 
//! ### Tensor
//! - Uses a tensor to store the error correction code
//! - Can handle a wide range of error patterns
//! 
//! ### Parallel
//! - Uses a parallel algorithm to encode and decode data
//! - Can handle a wide range of error patterns 
//! 
//! ### Adaptive Parallel
//! - Automatically adjusts the error correction code based on the error pattern
//! - Uses a parallel algorithm to encode and decode data
//! - Uses a lookup table to store the error correction code
//! - Can handle a wide range of error patterns
//! 
//! ### Parallel Tensor
//! - Uses a parallel algorithm to encode and decode data
//! - Uses a tensor to store the error correction code
//! - Can handle a wide range of error patterns 
//! 
//! ### Adaptive Parallel Tensor
//! - Automatically adjusts the error correction code based on the error pattern
//! - Uses a parallel algorithm to encode and decode data
//! - Uses a tensor to store the error correction code
//! - Can handle a wide range of error patterns

use std::fmt::Debug;
use std::sync::Arc;
use std::path::Path;

use crate::error::Result;
use crate::galois::GaloisField;
use crate::hardware::HardwareAccelerator;
use crate::xypher_grid::XypherGrid;

// Submodules
mod reed_solomon;
mod ldpc;
mod polar_code;
mod hamming;
mod bch;
mod reed_muller;
mod turbo;
mod convolutional;
mod fountain;

// Public exports
pub use reed_solomon::ReedSolomon;
pub use reed_solomon::TensorReedSolomon;
pub use reed_solomon::AdaptiveReedSolomon;
pub use ldpc::Ldpc;
pub use hamming::HammingCode;
pub use hamming::ExtendedHammingCode;
pub use bch::BchCode;
pub use bch::ExtendedBchCode;
pub use reed_muller::ReedMullerCode;
pub use turbo::TurboCode;
pub use convolutional::ConvolutionalCode;
pub use fountain::FountainCode;

/// Enum representing different types of error correction algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgorithmType {
    /// Reed-Solomon algorithm
    ReedSolomon,
    /// Tensor Reed-Solomon algorithm
    TensorReedSolomon,
    /// Adaptive Reed-Solomon algorithm
    AdaptiveReedSolomon,
    /// Low-Density Parity-Check algorithm
    Ldpc,
    /// Polar code algorithm
    PolarCode,
    /// Hamming code algorithm
    HammingCode,
    /// Extended Hamming code algorithm
    ExtendedHammingCode,
    /// BCH code algorithm
    BchCode,
    /// Extended BCH code algorithm
    ExtendedBchCode,
    /// Reed-Muller code algorithm
    ReedMullerCode,
    /// Turbo code algorithm
    Turbo,
    /// Convolutional code algorithm
    Convolutional,
    /// Fountain code algorithm
    Fountain,
    /// XypherGrid-accelerated BCH code algorithm
    XypherGridBch,
    /// XypherGrid-accelerated Reed-Solomon algorithm
    XypherGridReedSolomon,
    /// XypherGrid-accelerated Hamming code algorithm
    XypherGridHamming,
    /// XypherGrid-accelerated Reed-Muller code algorithm
    XypherGridReedMuller,
}

/// Trait defining the interface for error correction algorithms
pub trait ErrorCorrectionAlgorithm: Debug + Send + Sync {
    /// Returns the type of this algorithm
    fn algorithm_type(&self) -> AlgorithmType;
    
    /// Encodes the input data with error correction codes
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Decodes the input data and corrects errors if possible
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>>;
    
    /// Returns the maximum number of errors this algorithm can correct
    fn max_correctable_errors(&self) -> usize;
    
    /// Returns the overhead ratio (encoded_size / original_size)
    fn overhead_ratio(&self) -> f64;
    
    /// Generates lookup tables for the algorithm if applicable
    fn generate_lookup_tables(&self, path: &Path) -> Result<()>;
    
    /// Returns true if the algorithm can use hardware acceleration
    fn supports_hardware_acceleration(&self) -> bool;
    
    /// Sets the hardware accelerator to use for operations
    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>);
    
    /// Returns true if the algorithm can use precomputed tables from XypherGrid
    fn supports_xypher_grid(&self) -> bool {
        false
    }
    
    /// Sets the XypherGrid to use for precomputed tables
    fn set_xypher_grid(&mut self, _xypher_grid: Arc<XypherGrid>) {
        // Default implementation does nothing
    }
    
    /// Returns the name of the algorithm
    fn name(&self) -> &str;
}

/// Factory function to create an error correction algorithm based on type
pub fn create_algorithm(
    algorithm_type: AlgorithmType,
    galois_field: Arc<GaloisField>,
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
) -> Result<Box<dyn ErrorCorrectionAlgorithm>> {
    match algorithm_type {
        AlgorithmType::ReedSolomon => {
            let rs = ReedSolomon::new(galois_field.clone(), 255, 223, hardware_accelerator)?;
            Ok(Box::new(rs))
        },
        AlgorithmType::TensorReedSolomon => {
            let rs = TensorReedSolomon::new(galois_field.clone(), 255, 223, 2, hardware_accelerator)?;
            Ok(Box::new(rs))
        },
        AlgorithmType::AdaptiveReedSolomon => {
            let rs = AdaptiveReedSolomon::new(galois_field.clone(), 255, 223, 2, hardware_accelerator)?;
            Ok(Box::new(rs))
        },
        AlgorithmType::Ldpc => {
            let ldpc = Ldpc::new(hardware_accelerator)?;
            Ok(Box::new(ldpc))
        },
        AlgorithmType::PolarCode => {
            // let polar = PolarCode::new(hardware_accelerator)?;
            // Ok(Box::new(polar))
            unimplemented!()
        },
        AlgorithmType::HammingCode => {
            let hamming = HammingCode::new(hardware_accelerator)?;
            Ok(Box::new(hamming))
        },
        AlgorithmType::ExtendedHammingCode => {
            let hamming = ExtendedHammingCode::new(hardware_accelerator)?;
            Ok(Box::new(hamming))
        },
        AlgorithmType::BchCode => {
            let bch = BchCode::new(galois_field.clone(), 63, 45, 3, hardware_accelerator)?;
            Ok(Box::new(bch))
        },
        AlgorithmType::ExtendedBchCode => {
            let bch = ExtendedBchCode::new(galois_field.clone(), 64, 45, 4, hardware_accelerator)?;
            Ok(Box::new(bch))
        },
        AlgorithmType::ReedMullerCode => {
            let rm = ReedMullerCode::new(galois_field.clone(), 3, 7, hardware_accelerator)?;
            Ok(Box::new(rm))
        },
        AlgorithmType::Turbo => {
            let turbo = TurboCode::new(hardware_accelerator)?;
            Ok(Box::new(turbo))
        },
        AlgorithmType::Convolutional => {
            let conv = ConvolutionalCode::new(hardware_accelerator)?;
            Ok(Box::new(conv))
        },
        AlgorithmType::Fountain => {
            let fountain = FountainCode::new(hardware_accelerator)?;
            Ok(Box::new(fountain))
        },
        AlgorithmType::XypherGridBch => {
            // Create a BCH algorithm and initialize it with XypherGrid
            let mut bch = BchCode::new(galois_field.clone(), 63, 45, 3, hardware_accelerator)?;
            
            // Get XypherGrid and set it
            let xypher_grid = Arc::new(crate::xypher_grid::XypherGrid::new());
            xypher_grid.initialize_precomputed_tables()?;
            bch.set_xypher_grid(xypher_grid);
            
            Ok(Box::new(bch))
        },
        AlgorithmType::XypherGridReedSolomon => {
            // Create a Reed-Solomon algorithm and initialize it with XypherGrid
            let mut rs = ReedSolomon::new(galois_field.clone(), 255, 223, hardware_accelerator)?;
            
            // Get XypherGrid and set it
            let xypher_grid = Arc::new(crate::xypher_grid::XypherGrid::new());
            xypher_grid.initialize_precomputed_tables()?;
            rs.set_xypher_grid(xypher_grid);
            
            Ok(Box::new(rs))
        },
        AlgorithmType::XypherGridHamming => {
            // Create a Hamming algorithm and initialize it with XypherGrid
            let mut hamming = HammingCode::new(hardware_accelerator)?;
            
            // Get XypherGrid and set it
            let xypher_grid = Arc::new(crate::xypher_grid::XypherGrid::new());
            xypher_grid.initialize_precomputed_tables()?;
            hamming.set_xypher_grid(xypher_grid);
            
            Ok(Box::new(hamming))
        },
        AlgorithmType::XypherGridReedMuller => {
            // Create a Reed-Muller algorithm and initialize it with XypherGrid
            let mut rm = ReedMullerCode::new(galois_field.clone(), 3, 7, hardware_accelerator)?;
            
            // Get XypherGrid and set it
            let xypher_grid = Arc::new(crate::xypher_grid::XypherGrid::new());
            xypher_grid.initialize_precomputed_tables()?;
            rm.set_xypher_grid(xypher_grid);
            
            Ok(Box::new(rm))
        },
    }
}

/// Selects the best algorithm for the given data and error characteristics
pub fn select_best_algorithm(
    data_size: usize,
    error_rate: f64,
    burst_errors: bool,
    galois_field: Arc<GaloisField>,
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
) -> Result<Box<dyn ErrorCorrectionAlgorithm>> {
    // Check if hardware accelerator supports XypherGrid
    let use_xypher_grid = hardware_accelerator.supports_xypher_grid();
    
    // For large data with low error rate, use LDPC
    if data_size > 10000 && error_rate < 0.01 {
        return create_algorithm(AlgorithmType::Ldpc, galois_field, hardware_accelerator);
    }
    
    // For burst errors, use Reed-Solomon (possibly XypherGrid accelerated)
    if burst_errors {
        if use_xypher_grid {
            return create_algorithm(AlgorithmType::XypherGridReedSolomon, galois_field, hardware_accelerator);
        } else {
            return create_algorithm(AlgorithmType::ReedSolomon, galois_field, hardware_accelerator);
        }
    }
    
    // For high error rates, use Turbo codes
    if error_rate > 0.05 {
        return create_algorithm(AlgorithmType::Turbo, galois_field, hardware_accelerator);
    }
    
    // For streaming data with low error rate, use BCH codes (possibly XypherGrid accelerated)
    if data_size < 1000 && error_rate < 0.02 {
        if use_xypher_grid {
            return create_algorithm(AlgorithmType::XypherGridBch, galois_field, hardware_accelerator);
        } else {
            return create_algorithm(AlgorithmType::BchCode, galois_field, hardware_accelerator);
        }
    }
    
    // For very low error rates and short block lengths, use Hamming codes (possibly XypherGrid accelerated)
    if error_rate < 0.001 && data_size < 128 {
        if use_xypher_grid {
            return create_algorithm(AlgorithmType::XypherGridHamming, galois_field, hardware_accelerator);
        } else {
            return create_algorithm(AlgorithmType::HammingCode, galois_field, hardware_accelerator);
        }
    }
    
    // For very low error rates and short to medium block lengths, use Polar codes
    if error_rate < 0.001 && data_size < 5000 {
        return create_algorithm(AlgorithmType::PolarCode, galois_field, hardware_accelerator);
    }
    
    // Default to Adaptive Reed-Solomon
    create_algorithm(AlgorithmType::AdaptiveReedSolomon, galois_field, hardware_accelerator)
} 