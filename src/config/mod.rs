//! Configuration settings for the HyVERX system.
//!
//! This module defines the configuration options that control the behavior
//! of the HyVERX error correction system, including hardware acceleration
//! targets, algorithm selection, and performance tuning parameters.

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Hardware acceleration target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// Automatically select the best available hardware
    Auto,
    /// Use CPU only with AVX2 acceleration if available
    Cpu,
    /// Use CUDA acceleration on NVIDIA GPUs
    Cuda,
    /// Use OpenCL acceleration (Intel Xe, AMD, etc.)
    OpenCL,
    /// Use all available hardware in parallel
    All,
}

/// Configuration settings for the HyVERX system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Size of error correction code in bytes
    ecc_size: usize,
    /// Maximum size of error correction code in bytes
    max_ecc_size: usize,
    /// Number of dimensions for error correction
    dimensions: usize,
    /// Maximum data size in bytes
    max_data_size: usize,
    /// Hardware acceleration target
    hardware_target: HardwareTarget,
    /// Path to lookup tables
    table_path: PathBuf,
    /// Number of threads to use (0 for auto)
    threads: usize,
    /// Verbose output flag
    verbose: bool,
    /// Field polynomial for Galois field (default: 0x11D for GF(2^8))
    field_polynomial: u32,
    /// Whether to use neural-symbolic integration
    use_neural_symbolic: bool,
    /// Whether to use precomputed lookup tables
    use_lookup_tables: bool,
    /// Level of parallelism for multi-dimensional error correction
    parallelism_level: ParallelismLevel,
    /// Memory limit for operations in megabytes (0 for unlimited)
    memory_limit_mb: usize,
}

/// Level of parallelism for multi-dimensional error correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismLevel {
    /// No parallelism, single-threaded operation
    None,
    /// Threads only, no dimension splitting
    Threads,
    /// Split dimensions across threads
    Dimensions,
    /// Full parallelism with dimension splitting and nested parallelism
    Full,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ecc_size: 32,
            max_ecc_size: 256,
            dimensions: 16,
            max_data_size: 10000,
            hardware_target: HardwareTarget::Auto,
            table_path: PathBuf::from("./tables"),
            threads: 0, // Auto
            verbose: false,
            field_polynomial: 0x11D, // GF(2^8)
            use_neural_symbolic: true,
            use_lookup_tables: true,
            parallelism_level: ParallelismLevel::Full,
            memory_limit_mb: 0, // Unlimited
        }
    }
}

impl Config {
    /// Creates a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the size of error correction code in bytes.
    ///
    /// # Arguments
    ///
    /// * `ecc_size` - Size of error correction code in bytes
    ///
    /// # Returns
    ///
    /// Updated configuration with the new ECC size
    pub fn with_ecc_size(mut self, ecc_size: usize) -> Self {
        self.ecc_size = ecc_size;
        self
    }

    /// Sets the maximum size of error correction code in bytes.
    ///
    /// # Arguments
    ///
    /// * `max_ecc_size` - Maximum size of error correction code in bytes
    ///
    /// # Returns
    ///
    /// Updated configuration with the new maximum ECC size
    pub fn with_max_ecc_size(mut self, max_ecc_size: usize) -> Self {
        self.max_ecc_size = max_ecc_size;
        self
    }

    /// Sets the number of dimensions for error correction.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Number of dimensions for error correction
    ///
    /// # Returns
    ///
    /// Updated configuration with the new dimension count
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Sets the maximum data size in bytes.
    ///
    /// # Arguments
    ///
    /// * `max_data_size` - Maximum data size in bytes
    ///
    /// # Returns
    ///
    /// Updated configuration with the new maximum data size
    pub fn with_max_data_size(mut self, max_data_size: usize) -> Self {
        self.max_data_size = max_data_size;
        self
    }

    /// Sets the hardware acceleration target.
    ///
    /// # Arguments
    ///
    /// * `hardware_target` - Hardware acceleration target
    ///
    /// # Returns
    ///
    /// Updated configuration with the new hardware target
    pub fn with_hardware_target(mut self, hardware_target: HardwareTarget) -> Self {
        self.hardware_target = hardware_target;
        self
    }

    /// Sets the path to lookup tables.
    ///
    /// # Arguments
    ///
    /// * `table_path` - Path to lookup tables
    ///
    /// # Returns
    ///
    /// Updated configuration with the new table path
    pub fn with_table_path<P: AsRef<Path>>(mut self, table_path: P) -> Self {
        self.table_path = table_path.as_ref().to_path_buf();
        self
    }

    /// Sets the number of threads to use (0 for auto).
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads to use
    ///
    /// # Returns
    ///
    /// Updated configuration with the new thread count
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Sets the verbose output flag.
    ///
    /// # Arguments
    ///
    /// * `verbose` - Verbose output flag
    ///
    /// # Returns
    ///
    /// Updated configuration with the new verbose flag
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets the field polynomial for Galois field.
    ///
    /// # Arguments
    ///
    /// * `field_polynomial` - Field polynomial for Galois field
    ///
    /// # Returns
    ///
    /// Updated configuration with the new field polynomial
    pub fn with_field_polynomial(mut self, field_polynomial: u32) -> Self {
        self.field_polynomial = field_polynomial;
        self
    }

    /// Sets whether to use neural-symbolic integration.
    ///
    /// # Arguments
    ///
    /// * `use_neural_symbolic` - Whether to use neural-symbolic integration
    ///
    /// # Returns
    ///
    /// Updated configuration with the new neural-symbolic flag
    pub fn with_neural_symbolic(mut self, use_neural_symbolic: bool) -> Self {
        self.use_neural_symbolic = use_neural_symbolic;
        self
    }

    /// Sets whether to use precomputed lookup tables.
    ///
    /// # Arguments
    ///
    /// * `use_lookup_tables` - Whether to use precomputed lookup tables
    ///
    /// # Returns
    ///
    /// Updated configuration with the new lookup tables flag
    pub fn with_lookup_tables(mut self, use_lookup_tables: bool) -> Self {
        self.use_lookup_tables = use_lookup_tables;
        self
    }

    /// Sets the level of parallelism for multi-dimensional error correction.
    ///
    /// # Arguments
    ///
    /// * `parallelism_level` - Level of parallelism
    ///
    /// # Returns
    ///
    /// Updated configuration with the new parallelism level
    pub fn with_parallelism_level(mut self, parallelism_level: ParallelismLevel) -> Self {
        self.parallelism_level = parallelism_level;
        self
    }

    /// Sets the memory limit for operations in megabytes (0 for unlimited).
    ///
    /// # Arguments
    ///
    /// * `memory_limit_mb` - Memory limit in megabytes
    ///
    /// # Returns
    ///
    /// Updated configuration with the new memory limit
    pub fn with_memory_limit(mut self, memory_limit_mb: usize) -> Self {
        self.memory_limit_mb = memory_limit_mb;
        self
    }

    /// Returns the size of error correction code in bytes.
    pub fn ecc_size(&self) -> usize {
        self.ecc_size
    }

    /// Returns the maximum size of error correction code in bytes.
    pub fn max_ecc_size(&self) -> usize {
        self.max_ecc_size
    }

    /// Returns the number of dimensions for error correction.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the maximum data size in bytes.
    pub fn max_data_size(&self) -> usize {
        self.max_data_size
    }

    /// Returns the hardware acceleration target.
    pub fn hardware_target(&self) -> HardwareTarget {
        self.hardware_target
    }

    /// Returns the path to lookup tables.
    pub fn table_path(&self) -> &Path {
        &self.table_path
    }

    /// Returns the number of threads to use (0 for auto).
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Returns the maximum number of threads to use, accounting for auto settings.
    pub fn max_threads(&self) -> usize {
        if self.threads == 0 {
            // Auto: use available logical cores
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        } else {
            self.threads
        }
    }

    /// Returns whether verbose output is enabled.
    pub fn verbose(&self) -> bool {
        self.verbose
    }

    /// Returns the field polynomial for Galois field.
    pub fn field_polynomial(&self) -> u32 {
        self.field_polynomial
    }

    /// Returns whether to use neural-symbolic integration.
    pub fn use_neural_symbolic(&self) -> bool {
        self.use_neural_symbolic
    }

    /// Returns whether to use precomputed lookup tables.
    pub fn use_lookup_tables(&self) -> bool {
        self.use_lookup_tables
    }

    /// Returns the level of parallelism for multi-dimensional error correction.
    pub fn parallelism_level(&self) -> ParallelismLevel {
        self.parallelism_level
    }

    /// Returns the memory limit for operations in megabytes (0 for unlimited).
    pub fn memory_limit_mb(&self) -> usize {
        self.memory_limit_mb
    }

    /// Returns the memory limit in bytes (0 for unlimited).
    pub fn memory_limit_bytes(&self) -> usize {
        if self.memory_limit_mb == 0 {
            0 // Unlimited
        } else {
            self.memory_limit_mb * 1024 * 1024
        }
    }

    /// Validates the configuration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the configuration is valid, or an error if it's invalid
    pub fn validate(&self) -> crate::Result<()> {
        // Validate ECC size
        if self.ecc_size == 0 {
            return Err(crate::Error::InvalidConfiguration(
                "ECC size must be greater than zero".to_string(),
            ));
        }

        if self.ecc_size > self.max_ecc_size {
            return Err(crate::Error::InvalidConfiguration(
                format!(
                    "ECC size ({}) cannot be greater than maximum ECC size ({})",
                    self.ecc_size, self.max_ecc_size
                ),
            ));
        }

        // Validate dimensions
        if self.dimensions == 0 {
            return Err(crate::Error::InvalidConfiguration(
                "Dimensions must be greater than zero".to_string(),
            ));
        }

        // Validate field polynomial
        if self.field_polynomial == 0 {
            return Err(crate::Error::InvalidConfiguration(
                "Field polynomial must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.ecc_size, 32);
        assert_eq!(config.dimensions, 16);
        assert_eq!(config.field_polynomial, 0x11D);
        assert!(matches!(config.hardware_target, HardwareTarget::Auto));
    }

    #[test]
    fn test_config_builder() {
        let config = Config::new()
            .with_ecc_size(64)
            .with_dimensions(8)
            .with_hardware_target(HardwareTarget::Cpu)
            .with_field_polynomial(0x1053)
            .with_threads(4)
            .with_verbose(true);

        assert_eq!(config.ecc_size(), 64);
        assert_eq!(config.dimensions(), 8);
        assert_eq!(config.field_polynomial(), 0x1053);
        assert!(matches!(config.hardware_target(), HardwareTarget::Cpu));
        assert_eq!(config.threads(), 4);
        assert_eq!(config.max_threads(), 4);
        assert!(config.verbose());
    }

    #[test]
    fn test_auto_threads() {
        let config = Config::new().with_threads(0);
        assert_eq!(config.threads(), 0);
        assert!(config.max_threads() > 0);
    }

    #[test]
    fn test_memory_limit() {
        let config = Config::new().with_memory_limit(1024);
        assert_eq!(config.memory_limit_mb(), 1024);
        assert_eq!(config.memory_limit_bytes(), 1024 * 1024 * 1024);

        let unlimited = Config::new().with_memory_limit(0);
        assert_eq!(unlimited.memory_limit_mb(), 0);
        assert_eq!(unlimited.memory_limit_bytes(), 0);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = Config::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_ecc_size() {
        let config = Config::new().with_ecc_size(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_dimensions() {
        let config = Config::new().with_dimensions(0);
        assert!(config.validate().is_err());
    }
}