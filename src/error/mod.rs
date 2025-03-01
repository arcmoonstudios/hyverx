//! Error handling for the HyVERX library.
//!
//! This module defines the error types used throughout the HyVERX system.

use std::path::PathBuf;
use thiserror::Error;

/// Result type used throughout the HyVERX system.
pub type Result<T> = std::result::Result<T, Error>;

/// Error enum for the HyVERX system.
#[derive(Error, Debug)]
pub enum Error {
    /// Error during Galois field operations
    #[error("Galois field error: {0}")]
    GaloisField(String),

    /// Error with hardware acceleration
    #[error("Hardware acceleration error: {0}")]
    HardwareAcceleration(String),

    /// Hardware is unavailable
    #[error("Hardware unavailable: {0}")]
    HardwareUnavailable(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Error during Reed-Solomon encoding/decoding
    #[error("Reed-Solomon error: {0}")]
    ReedSolomon(String),

    /// Error during LDPC encoding/decoding
    #[error("LDPC error: {0}")]
    Ldpc(String),

    /// Error during Turbo encoding/decoding
    #[error("Turbo code error: {0}")]
    TurboCode(String),

    /// Error during Convolutional encoding/decoding
    #[error("Convolutional code error: {0}")]
    ConvolutionalCode(String),

    /// Error in neural-symbolic processing
    #[error("Neural-symbolic error: {0}")]
    NeuralSymbolic(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    Decoding(String),

    /// Too many errors to correct
    #[error("Too many errors to correct: {detected} detected, maximum is {correctable}")]
    TooManyErrors {
        /// Number of errors detected
        detected: usize,
        /// Maximum number of correctable errors
        correctable: usize,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to initialize hardware
    #[error("Hardware initialization failed: {0}")]
    HardwareInitialization(String),

    /// Error in CUDA operations
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Error in OpenCL operations
    #[cfg(feature = "opencl")]
    #[error("OpenCL error: {0}")]
    OpenCL(String),

    /// Error with lookup tables
    #[error("Lookup table error: {0}")]
    LookupTable(String),

    /// Missing lookup table file
    #[error("Missing lookup table file: {0}")]
    MissingLookupTable(PathBuf),

    /// Invalid input data
    #[error("Invalid input data: {0}")]
    InvalidInput(String),

    /// Algorithm selection error
    #[error("Algorithm selection error: {0}")]
    AlgorithmSelection(String),

    /// Thread pool error
    #[error("Thread pool error: {0}")]
    ThreadPool(String),

    /// Thread pool shutdown error
    #[error("Thread pool shutdown: {0}")]
    ThreadPoolShutdown(String),

    /// Queue full error
    #[error("Queue full: {0}")]
    QueueFull(String),

    /// Error during serialization/deserialization
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Error during binary serialization/deserialization
    #[error("Binary serialization error: {0}")]
    BinarySerialization(#[from] bincode::Error),

    /// Dimension mismatch in multi-dimensional operations
    #[error("Dimension mismatch: expected {expected} dimensions, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions
        expected: usize,
        /// Actual number of dimensions
        actual: usize,
    },

    /// Insufficient memory for operation
    #[error("Insufficient memory: need {needed} bytes, available {available} bytes")]
    InsufficientMemory {
        /// Memory needed for the operation
        needed: usize,
        /// Memory available
        available: usize,
    },

    /// Generic error with a message
    #[error("{0}")]
    Generic(String),
    
    /// Errors from other sources
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

// Conversion from hardware-specific errors
#[cfg(feature = "cuda")]
impl From<rustacuda::error::CudaError> for Error {
    fn from(err: rustacuda::error::CudaError) -> Self {
        Error::Cuda(err.to_string())
    }
}

#[cfg(feature = "opencl")]
impl From<opencl3::error_codes::ClError> for Error {
    fn from(err: opencl3::error_codes::ClError) -> Self {
        Error::OpenCL(err.to_string())
    }
}

// Add the new implementation
impl From<Box<dyn std::error::Error>> for Error {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        Error::Internal(format!("External error: {}", err))
    }
}