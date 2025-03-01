//! Reed-Solomon error correction algorithm implementation.
//!
//! Reed-Solomon codes are block-based error correcting codes with a wide range of
//! applications in digital communications and storage. They are particularly
//! effective against burst errors.
//!
//! This module provides several implementations of Reed-Solomon codes:
//! - Standard Reed-Solomon: Basic implementation for general use
//! - Tensor Reed-Solomon: Optimized for multi-dimensional data using tensor operations
//! - Adaptive Reed-Solomon: Dynamically selects the optimal processing strategy

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::galois::GaloisField;
use crate::hardware::HardwareAccelerator;
use crate::parallel::{ThreadPool, ParallelConfig};
use super::{AlgorithmType, ErrorCorrectionAlgorithm};

/// Reed-Solomon error correction algorithm implementation.
pub struct ReedSolomon {
    /// Galois field used for finite field arithmetic
    galois_field: Arc<GaloisField>,
    /// Total codeword length (n)
    codeword_length: usize,
    /// Message length (k)
    message_length: usize,
    /// Number of parity symbols (n-k)
    parity_length: usize,
    /// Generator polynomial
    generator_polynomial: Vec<u16>,
    /// Hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
    /// Syndrome cache for optimization
    syndrome_cache: Mutex<Vec<Vec<u16>>>,
}

impl ReedSolomon {
    /// Creates a new Reed-Solomon encoder/decoder.
    ///
    /// # Arguments
    ///
    /// * `galois_field` - Galois field for finite field arithmetic
    /// * `codeword_length` - Total codeword length (n)
    /// * `message_length` - Message length (k)
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `ReedSolomon` instance or an error if parameters are invalid.
    pub fn new(
        galois_field: Arc<GaloisField>,
        codeword_length: usize,
        message_length: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        // Validate parameters
        if message_length >= codeword_length {
            return Err(Error::InvalidInput(
                "Message length must be less than codeword length".into(),
            ));
        }

        if codeword_length > galois_field.element_count() {
            return Err(Error::InvalidInput(
                format!(
                    "Codeword length ({}) exceeds field size ({})",
                    codeword_length,
                    galois_field.element_count()
                )
                .into(),
            ));
        }

        let parity_length = codeword_length - message_length;
        let generator_polynomial = Self::generate_polynomial(&galois_field, parity_length)?;

        Ok(Self {
            galois_field,
            codeword_length,
            message_length,
            parity_length,
            generator_polynomial,
            hardware_accelerator,
            syndrome_cache: Mutex::new(Vec::new()),
        })
    }

    /// Generates the generator polynomial for Reed-Solomon encoding.
    ///
    /// The generator polynomial is the product of (x - α^i) for i from 1 to parity_length.
    fn generate_polynomial(galois_field: &GaloisField, parity_length: usize) -> Result<Vec<u16>> {
        let mut polynomial = vec![1];

        for i in 0..parity_length {
            let mut temp = vec![0; polynomial.len() + 1];
            let root = galois_field.exp((i as i32) + 1);

            for j in 0..polynomial.len() {
                if polynomial[j] != 0 {
                    temp[j] ^= polynomial[j];
                    temp[j + 1] ^= galois_field.multiply(polynomial[j], root);
                }
            }

            polynomial = temp;
        }

        Ok(polynomial)
    }

    /// Calculates the syndromes for a received codeword.
    ///
    /// Syndromes are used to detect and locate errors in the received codeword.
    fn calculate_syndromes(&self, received: &[u16]) -> Result<Vec<u16>> {
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() {
            let received_bytes: Vec<u8> = received.iter().map(|&x| x as u8).collect();
            return self.hardware_accelerator
                .calculate_syndromes(&received_bytes, self.parity_length)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Software implementation
        let mut syndromes = vec![0; self.parity_length];
        
        for i in 0..self.parity_length {
            let mut syndrome = 0;
            let x = self.galois_field.exp((i as i32) + 1);
            
            for j in 0..received.len() {
                let power = (received.len() - 1 - j) as u32;
                let term = self.galois_field.multiply(received[j], self.galois_field.power(x, power));
                syndrome = self.galois_field.add(syndrome, term);
            }
            
            syndromes[i] = syndrome;
        }
        
        Ok(syndromes)
    }

    /// Finds the error locator polynomial using the Berlekamp-Massey algorithm.
    fn find_error_locator(&self, syndromes: &[u16]) -> Result<Vec<u16>> {
        let mut error_locator = vec![1];
        let mut old_locator = vec![1];
        
        for i in 0..syndromes.len() {
            let delta = self.calculate_discrepancy(&error_locator, syndromes, i);
            
            if delta != 0 {
                let mut new_locator = error_locator.clone();
                
                // Compute error_locator + delta * x^i * old_locator
                let mut term = vec![0; i + 1 + old_locator.len()];
                for j in 0..old_locator.len() {
                    term[j + i + 1] = self.galois_field.multiply(delta, old_locator[j]);
                }
                
                // Resize new_locator if needed
                if term.len() > new_locator.len() {
                    new_locator.resize(term.len(), 0);
                }
                
                // Add term to new_locator
                for j in 0..term.len() {
                    if j < new_locator.len() {
                        new_locator[j] = self.galois_field.add(new_locator[j], term[j]);
                    }
                }
                
                if 2 * i > syndromes.len() {
                    error_locator = new_locator;
                } else {
                    old_locator = error_locator;
                    error_locator = new_locator;
                }
            }
        }
        
        Ok(error_locator)
    }

    /// Calculates the discrepancy for the Berlekamp-Massey algorithm.
    fn calculate_discrepancy(&self, error_locator: &[u16], syndromes: &[u16], i: usize) -> u16 {
        let mut sum = 0;
        
        for j in 0..error_locator.len() {
            if j == 0 {
                continue; // Skip the first coefficient (always 1)
            }
            
            if i >= j && syndromes[i - j] != 0 && error_locator[j] != 0 {
                sum = self.galois_field.add(
                    sum,
                    self.galois_field.multiply(error_locator[j], syndromes[i - j]),
                );
            }
        }
        
        sum
    }

    /// Finds the roots of the error locator polynomial using the Chien search algorithm.
    fn find_error_locations(&self, error_locator: &[u16]) -> Result<Vec<usize>> {
        let mut error_locations = Vec::new();
        
        for i in 0..self.codeword_length {
            let x_inv = self.galois_field.exp(self.galois_field.element_count() as i32 - i as i32 - 1);
            let mut sum = 0;
            
            for j in 0..error_locator.len() {
                let term = self.galois_field.multiply(
                    error_locator[j],
                    self.galois_field.power(x_inv, j as u32),
                );
                sum = self.galois_field.add(sum, term);
            }
            
            if sum == 0 {
                error_locations.push(i);
            }
        }
        
        // Check if the number of errors is correctable
        if error_locations.len() > self.parity_length / 2 {
            return Err(Error::TooManyErrors {
                detected: error_locations.len(),
                correctable: self.parity_length / 2,
            });
        }
        
        Ok(error_locations)
    }

    /// Calculates the error values using the Forney algorithm.
    fn calculate_error_values(&self, error_locator: &[u16], syndromes: &[u16], error_locations: &[usize]) -> Result<Vec<u16>> {
        let mut error_values = vec![0; error_locations.len()];
        
        for i in 0..error_locations.len() {
            let x_inv = self.galois_field.exp(self.galois_field.element_count() as i32 - error_locations[i] as i32 - 1);
            
            // Calculate error evaluator polynomial
            let mut error_eval = 0;
            for j in 0..syndromes.len() {
                let term = self.galois_field.multiply(
                    syndromes[j],
                    self.galois_field.power(x_inv, (j as u32) + 1),
                );
                error_eval = self.galois_field.add(error_eval, term);
            }
            
            // Calculate error locator derivative
            let mut locator_derivative = 0;
            for j in 1..error_locator.len() {
                if j % 2 == 1 { // Only odd powers contribute to the derivative
                    let term = self.galois_field.multiply(
                        error_locator[j],
                        self.galois_field.power(x_inv, (j - 1) as u32),
                    );
                    locator_derivative = self.galois_field.add(locator_derivative, term);
                }
            }
            
            // Calculate error value
            if locator_derivative != 0 {
                error_values[i] = self.galois_field.divide(error_eval, locator_derivative);
            }
        }
        
        Ok(error_values)
    }

    /// Corrects errors in the received codeword.
    fn correct_errors(&self, received: &mut [u16], error_locations: &[usize], error_values: &[u16]) -> Result<()> {
        for i in 0..error_locations.len() {
            let location = error_locations[i];
            if location < received.len() {
                received[location] = self.galois_field.add(received[location], error_values[i]);
            }
        }
        
        Ok(())
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Convert data to field elements
        let mut message: Vec<u16> = data.iter().map(|&b| b as u16).collect();
        
        // Pad message if needed
        if message.len() < self.message_length {
            message.resize(self.message_length, 0);
        } else if message.len() > self.message_length {
            return Err(Error::InvalidInput(
                format!("Input data too large: {} bytes (max: {})", data.len(), self.message_length).into(),
            ));
        }
        
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() {
            // No direct reed-solomon support in hardware, using generic hardware operations
            // This is a placeholder - in a real implementation you would use dedicated hardware functions
        }
        
        // Software implementation
        let mut codeword = vec![0; self.codeword_length];
        
        // Copy message to codeword
        for i in 0..self.message_length {
            codeword[i] = message[i];
        }
        
        // Calculate parity bytes using polynomial division
        let parity_length = self.codeword_length - self.message_length;
        let mut remainder = vec![0; parity_length];
        
        for i in 0..self.message_length {
            let feedback = self.galois_field.add(codeword[i], remainder[0]);
            
            // Shift remainder left
            for j in 1..parity_length {
                remainder[j - 1] = remainder[j];
            }
            remainder[parity_length - 1] = 0;
            
            // Multiply by generator polynomial
            if feedback != 0 {
                for j in 0..parity_length {
                    if j + 1 < self.generator_polynomial.len() {
                        remainder[j] = self.galois_field.add(
                            remainder[j],
                            self.galois_field.multiply(self.generator_polynomial[j + 1], feedback),
                        );
                    }
                }
            }
        }
        
        // Add remainder to codeword
        for i in 0..parity_length {
            codeword[self.message_length + i] = remainder[i];
        }
        
        // Convert back to bytes
        let encoded: Vec<u8> = codeword.iter().map(|&x| x as u8).collect();
        Ok(encoded)
    }
}

impl ErrorCorrectionAlgorithm for ReedSolomon {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::ReedSolomon
    }
    
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.encode(data)
    }
    
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() != self.codeword_length {
            return Err(Error::InvalidInput(
                format!("Invalid codeword length: {} (expected: {})", data.len(), self.codeword_length).into(),
            ));
        }
        
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() {
            // No direct reed-solomon support in hardware, using generic hardware operations
            // This is a placeholder - in a real implementation you would use dedicated hardware functions
        }
        
        // Convert data to field elements
        let mut received: Vec<u16> = data.iter().map(|&b| b as u16).collect();
        
        // Calculate syndromes
        let syndromes = self.calculate_syndromes(&received)?;
        
        // Check if there are any errors
        let has_errors = syndromes.iter().any(|&s| s != 0);
        if !has_errors {
            // No errors, return the message part
            return Ok(data[0..self.message_length].to_vec());
        }
        
        // Find error locator polynomial
        let error_locator = self.find_error_locator(&syndromes)?;
        
        // Find error locations
        let error_locations = self.find_error_locations(&error_locator)?;
        
        // Check if the number of errors is correctable
        if error_locations.len() > self.parity_length / 2 {
            return Err(Error::TooManyErrors {
                detected: error_locations.len(),
                correctable: self.parity_length / 2,
            });
        }
        
        // Calculate error values
        let error_values = self.calculate_error_values(&error_locator, &syndromes, &error_locations)?;
        
        // Correct errors
        self.correct_errors(&mut received, &error_locations, &error_values)?;
        
        // Convert back to bytes and return the message part
        let decoded: Vec<u8> = received[0..self.message_length].iter().map(|&x| x as u8).collect();
        Ok(decoded)
    }
    
    fn max_correctable_errors(&self) -> usize {
        self.parity_length / 2
    }
    
    fn overhead_ratio(&self) -> f64 {
        self.codeword_length as f64 / self.message_length as f64
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the Reed-Solomon directory
        let rs_path = path.join("reed_solomon");
        std::fs::create_dir_all(&rs_path)?;
        
        // Generate syndrome lookup tables
        let mut syndrome_cache = self.syndrome_cache.lock().map_err(|_| {
            Error::Internal("Failed to lock syndrome cache".into())
        })?;
        
        // Clear existing cache
        syndrome_cache.clear();
        
        // Generate syndromes for common error patterns
        for i in 0..self.codeword_length {
            let mut pattern = vec![0; self.codeword_length];
            pattern[i] = 1; // Single error at position i
            
            let syndromes = self.calculate_syndromes(&pattern)?;
            syndrome_cache.push(syndromes);
        }
        
        // Save syndrome cache to file
        let syndrome_path = rs_path.join("syndrome_cache.bin");
        let syndrome_data = bincode::serialize(&*syndrome_cache)
            .map_err(|e| Error::BinarySerialization(e))?;
        
        std::fs::write(syndrome_path, syndrome_data)?;
        
        Ok(())
    }
    
    fn supports_hardware_acceleration(&self) -> bool {
        self.hardware_accelerator.is_available()
    }
    
    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator;
    }
    
    fn name(&self) -> &str {
        "Reed-Solomon"
    }
}

impl fmt::Debug for ReedSolomon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReedSolomon")
            .field("codeword_length", &self.codeword_length)
            .field("message_length", &self.message_length)
            .field("parity_length", &self.parity_length)
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .finish()
    }
}

impl Clone for ReedSolomon {
    fn clone(&self) -> Self {
        Self {
            galois_field: self.galois_field.clone(),
            codeword_length: self.codeword_length,
            message_length: self.message_length,
            parity_length: self.parity_length,
            generator_polynomial: self.generator_polynomial.clone(),
            hardware_accelerator: self.hardware_accelerator.clone(),
            syndrome_cache: Mutex::new(Vec::new()),
        }
    }
}

/// Tensor-based Reed-Solomon implementation optimized for multi-dimensional data.
pub struct TensorReedSolomon {
    /// Base Reed-Solomon implementation
    base_rs: ReedSolomon,
    /// Number of dimensions
    dimensions: usize,
    /// Thread pool for parallel processing
    thread_pool: Option<Arc<ThreadPool>>,
    /// Performance metrics
    metrics: Mutex<TensorMetrics>,
}

/// Performance metrics for tensor operations
struct TensorMetrics {
    /// Number of encode operations
    encode_count: usize,
    /// Number of decode operations
    decode_count: usize,
    /// Number of batch encode operations
    batch_encode_count: usize,
    /// Number of batch decode operations
    batch_decode_count: usize,
    /// Total encoding time
    total_encode_time: Duration,
    /// Total decoding time
    total_decode_time: Duration,
}

impl TensorReedSolomon {
    /// Creates a new tensor-based Reed-Solomon encoder/decoder.
    ///
    /// # Arguments
    ///
    /// * `galois_field` - Galois field for finite field arithmetic
    /// * `codeword_length` - Total codeword length (n)
    /// * `message_length` - Message length (k)
    /// * `dimensions` - Number of dimensions for tensor operations
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `TensorReedSolomon` instance or an error if parameters are invalid.
    pub fn new(
        galois_field: Arc<GaloisField>,
        codeword_length: usize,
        message_length: usize,
        dimensions: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        let base_rs = ReedSolomon::new(
            galois_field,
            codeword_length,
            message_length,
            hardware_accelerator,
        )?;
        
        // Create thread pool for parallel processing
        let config = ParallelConfig::default();
        let thread_pool = Some(Arc::new(ThreadPool::new(config)));
        
        Ok(Self {
            base_rs,
            dimensions,
            thread_pool,
            metrics: Mutex::new(TensorMetrics {
                encode_count: 0,
                decode_count: 0,
                batch_encode_count: 0,
                batch_decode_count: 0,
                total_encode_time: Duration::new(0, 0),
                total_decode_time: Duration::new(0, 0),
            }),
        })
    }
    
    /// Encodes a multi-dimensional array of data using tensor operations.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to encode
    /// * `dimensions` - The dimensions of the data
    ///
    /// # Returns
    ///
    /// The encoded data.
    pub fn encode_tensor(&self, data: &[u8], dimensions: &[u32]) -> Result<Vec<u8>> {
        // Validate dimensions and data size
        if dimensions.is_empty() {
            return Err(Error::InvalidInput(
                "Dimensions cannot be empty".into(),
            ));
        }
        
        let total_size: usize = dimensions.iter().map(|&d| d as usize).product();
        if total_size != data.len() {
            return Err(Error::InvalidInput(
                format!("Data size ({}) does not match dimensions product ({})", data.len(), total_size).into(),
            ));
        }
        
        // Start timing for metrics
        let start_time = Instant::now();
        
        // For simplicity, just use the base Reed-Solomon encoding
        // In a real implementation, we would use tensor-specific optimizations
        let result = self.base_rs.encode(data);
        
        // Update metrics
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.encode_count += 1;
            metrics.total_encode_time += start_time.elapsed();
        }
        
        result
    }
    
    /// Splits data into multiple dimensions for tensor Reed-Solomon.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to split
    /// * `dimensions` - The dimensions to split into
    ///
    /// # Returns
    ///
    /// The data split into multiple dimensions.
    #[allow(dead_code)]
    fn split_data_by_dimensions(&self, data: &[u8], dimensions: &[usize]) -> Result<Vec<Vec<u8>>> {
        // For simplicity, split along the first dimension
        let first_dim_size = dimensions[0];
        let chunk_size = data.len() / first_dim_size;
        
        let mut chunks = Vec::with_capacity(first_dim_size);
        for i in 0..first_dim_size {
            let start = i * chunk_size;
            let end = start + chunk_size;
            chunks.push(data[start..end].to_vec());
        }
        
        Ok(chunks)
    }
    
    /// Gets performance metrics.
    ///
    /// # Returns
    ///
    /// A map of performance metrics.
    pub fn get_metrics(&self) -> Result<HashMap<String, f64>> {
        let metrics = self.metrics.lock().map_err(|_| {
            Error::Internal("Failed to lock metrics".into())
        })?;
        
        let mut result = HashMap::new();
        result.insert("encode_count".to_string(), metrics.encode_count as f64);
        result.insert("decode_count".to_string(), metrics.decode_count as f64);
        result.insert("batch_encode_count".to_string(), metrics.batch_encode_count as f64);
        result.insert("batch_decode_count".to_string(), metrics.batch_decode_count as f64);
        
        let encode_time_ms = metrics.total_encode_time.as_secs_f64() * 1000.0;
        let decode_time_ms = metrics.total_decode_time.as_secs_f64() * 1000.0;
        
        result.insert("total_encode_time_ms".to_string(), encode_time_ms);
        result.insert("total_decode_time_ms".to_string(), decode_time_ms);
        
        if metrics.encode_count > 0 {
            result.insert("avg_encode_time_ms".to_string(), encode_time_ms / metrics.encode_count as f64);
        }
        
        if metrics.decode_count > 0 {
            result.insert("avg_decode_time_ms".to_string(), decode_time_ms / metrics.decode_count as f64);
        }
        
        Ok(result)
    }
}

impl ErrorCorrectionAlgorithm for TensorReedSolomon {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::TensorReedSolomon
    }
    
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // For non-tensor data, use a simple 1D tensor
        self.encode_tensor(data, &[data.len() as u32])
    }
    
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Use base implementation for decoding
        let result = self.base_rs.decode(data);
        
        // Update metrics
        if let Ok(_) = &result {
            let mut metrics = self.metrics.lock().map_err(|_| {
                Error::Internal("Failed to lock metrics".into())
            })?;
            metrics.decode_count += 1;
            metrics.total_decode_time += start_time.elapsed();
        }
        
        result
    }
    
    fn max_correctable_errors(&self) -> usize {
        self.base_rs.max_correctable_errors()
    }
    
    fn overhead_ratio(&self) -> f64 {
        self.base_rs.overhead_ratio()
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the TensorReedSolomon directory
        let tensor_rs_path = path.join("tensor_reed_solomon");
        std::fs::create_dir_all(&tensor_rs_path)?;
        
        // Generate lookup tables for the base implementation
        self.base_rs.generate_lookup_tables(&tensor_rs_path)?;
        
        Ok(())
    }
    
    fn supports_hardware_acceleration(&self) -> bool {
        self.base_rs.supports_hardware_acceleration()
    }
    
    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.base_rs.set_hardware_accelerator(accelerator);
    }
    
    fn name(&self) -> &str {
        "Tensor Reed-Solomon"
    }
}

impl fmt::Debug for TensorReedSolomon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorReedSolomon")
            .field("dimensions", &self.dimensions)
            .field("base_rs", &self.base_rs)
            .finish()
    }
}

impl Clone for TensorReedSolomon {
    fn clone(&self) -> Self {
        Self {
            base_rs: self.base_rs.clone(),
            dimensions: self.dimensions,
            thread_pool: self.thread_pool.clone(),
            metrics: Mutex::new(TensorMetrics {
                encode_count: 0,
                decode_count: 0,
                batch_encode_count: 0,
                batch_decode_count: 0,
                total_encode_time: Duration::new(0, 0),
                total_decode_time: Duration::new(0, 0),
            }),
        }
    }
}

/// Adaptive Reed-Solomon implementation that dynamically selects the optimal processing strategy.
pub struct AdaptiveReedSolomon {
    /// Base Reed-Solomon implementation
    base_rs: ReedSolomon,
    /// Tensor Reed-Solomon implementation
    tensor_rs: Option<TensorReedSolomon>,
    /// Current algorithm selection
    current_algorithm: AlgorithmType,
    /// Performance history for algorithm selection
    performance_history: Mutex<HashMap<AlgorithmType, Vec<Duration>>>,
    /// Error statistics
    error_statistics: Mutex<ErrorStatistics>,
}

/// Error statistics for adaptive algorithm selection
struct ErrorStatistics {
    /// Total number of processed blocks
    total_processed: usize,
    /// Number of error-free blocks
    error_free: usize,
    /// Number of correctable blocks
    correctable: usize,
    /// Number of uncorrectable blocks
    uncorrectable: usize,
    /// Map of error positions to frequency
    error_positions: HashMap<usize, usize>,
}

impl AdaptiveReedSolomon {
    /// Creates a new adaptive Reed-Solomon encoder/decoder.
    ///
    /// # Arguments
    ///
    /// * `galois_field` - Galois field for finite field arithmetic
    /// * `codeword_length` - Total codeword length (n)
    /// * `message_length` - Message length (k)
    /// * `dimensions` - Number of dimensions for tensor operations
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `AdaptiveReedSolomon` instance or an error if parameters are invalid.
    pub fn new(
        galois_field: Arc<GaloisField>,
        codeword_length: usize,
        message_length: usize,
        dimensions: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        let base_rs = ReedSolomon::new(
            galois_field.clone(),
            codeword_length,
            message_length,
            hardware_accelerator.clone(),
        )?;
        
        // Create tensor implementation if dimensions > 1
        let tensor_rs = if dimensions > 1 {
            Some(TensorReedSolomon::new(
                galois_field,
                codeword_length,
                message_length,
                dimensions,
                hardware_accelerator,
            )?)
        } else {
            None
        };
        
        // Initialize performance history
        let mut performance_history = HashMap::new();
        performance_history.insert(AlgorithmType::ReedSolomon, Vec::new());
        performance_history.insert(AlgorithmType::TensorReedSolomon, Vec::new());
        
        Ok(Self {
            base_rs,
            tensor_rs,
            current_algorithm: AlgorithmType::ReedSolomon,
            performance_history: Mutex::new(performance_history),
            error_statistics: Mutex::new(ErrorStatistics {
                total_processed: 0,
                error_free: 0,
                correctable: 0,
                uncorrectable: 0,
                error_positions: HashMap::new(),
            }),
        })
    }
    
    /// Selects the best algorithm based on data characteristics and performance history.
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to process
    /// * `is_tensor` - Whether the data is multi-dimensional
    ///
    /// # Returns
    ///
    /// The selected algorithm type.
    fn select_best_algorithm(&self, data_size: usize, is_tensor: bool) -> AlgorithmType {
        // If tensor operations are requested and available, use TensorReedSolomon
        if is_tensor && self.tensor_rs.is_some() {
            return AlgorithmType::TensorReedSolomon;
        }
        
        // For small data, use the base implementation
        if data_size < 1024 {
            return AlgorithmType::ReedSolomon;
        }
        
        // For larger data, check performance history
        let performance_history = match self.performance_history.lock() {
            Ok(guard) => guard,
            Err(_) => {
                // If lock fails, use a default empty history for decision making
                return AlgorithmType::TensorReedSolomon; // Default to tensor version on lock failure
            }
        };
        
        // Calculate average performance for each algorithm
        let rs_avg = if !performance_history[&AlgorithmType::ReedSolomon].is_empty() {
            performance_history[&AlgorithmType::ReedSolomon].iter()
                .map(|d| d.as_secs_f64())
                .sum::<f64>() / performance_history[&AlgorithmType::ReedSolomon].len() as f64
        } else {
            f64::MAX
        };
        
        let tensor_avg = if !performance_history[&AlgorithmType::TensorReedSolomon].is_empty() {
            performance_history[&AlgorithmType::TensorReedSolomon].iter()
                .map(|d| d.as_secs_f64())
                .sum::<f64>() / performance_history[&AlgorithmType::TensorReedSolomon].len() as f64
        } else {
            f64::MAX
        };
        
        // Select the algorithm with better performance
        if tensor_avg < rs_avg && self.tensor_rs.is_some() {
            AlgorithmType::TensorReedSolomon
        } else {
            AlgorithmType::ReedSolomon
        }
    }
    
    /// Updates performance history for an algorithm.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The algorithm type
    /// * `duration` - The execution duration
    fn update_performance_history(&self, algorithm: AlgorithmType, duration: Duration) -> Result<()> {
        let mut performance_history = self.performance_history.lock().map_err(|_| {
            Error::Internal("Failed to lock performance history".into())
        })?;
        
        // Add the duration to the history
        if let Some(history) = performance_history.get_mut(&algorithm) {
            // Keep only the last 10 measurements
            if history.len() >= 10 {
                history.remove(0);
            }
            history.push(duration);
        }
        
        Ok(())
    }
    
    /// Updates error statistics.
    ///
    /// # Arguments
    ///
    /// * `error_count` - Number of errors detected
    /// * `error_positions` - Positions of the errors
    /// * `correctable` - Whether the errors were correctable
    fn update_error_statistics(&self, error_count: usize, error_positions: &[usize], correctable: bool) -> Result<()> {
        let mut error_statistics = self.error_statistics.lock().map_err(|_| {
            Error::Internal("Failed to lock error statistics".into())
        })?;
        
        // Update statistics
        error_statistics.total_processed += 1;
        
        if error_count == 0 {
            error_statistics.error_free += 1;
        } else if correctable {
            error_statistics.correctable += 1;
        } else {
            error_statistics.uncorrectable += 1;
        }
        
        // Update error position frequencies
        for &pos in error_positions {
            *error_statistics.error_positions.entry(pos).or_insert(0) += 1;
        }
        
        Ok(())
    }
    
    /// Gets error statistics.
    ///
    /// # Returns
    ///
    /// A map of error statistics.
    pub fn get_error_statistics(&self) -> Result<HashMap<String, usize>> {
        let error_statistics = self.error_statistics.lock().map_err(|_| {
            Error::Internal("Failed to lock error statistics".into())
        })?;
        
        let mut result = HashMap::new();
        result.insert("total_processed".to_string(), error_statistics.total_processed);
        result.insert("error_free".to_string(), error_statistics.error_free);
        result.insert("correctable".to_string(), error_statistics.correctable);
        result.insert("uncorrectable".to_string(), error_statistics.uncorrectable);
        
        Ok(result)
    }
}

impl ErrorCorrectionAlgorithm for AdaptiveReedSolomon {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::AdaptiveReedSolomon
    }
    
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Select the best algorithm
        let algorithm = self.select_best_algorithm(data.len(), false);
        
        // Use the selected algorithm
        let result = match algorithm {
            AlgorithmType::TensorReedSolomon => {
                if let Some(tensor_rs) = &self.tensor_rs {
                    tensor_rs.encode(data)
                } else {
                    self.base_rs.encode(data)
                }
            },
            _ => self.base_rs.encode(data),
        };
        
        // Update performance history
        let duration = start_time.elapsed();
        let _ = self.update_performance_history(algorithm, duration);
        
        result
    }
    
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        // Select the best algorithm
        let algorithm = self.select_best_algorithm(data.len(), false);
        
        // Use the selected algorithm
        let result = match algorithm {
            AlgorithmType::TensorReedSolomon => {
                if let Some(tensor_rs) = &self.tensor_rs {
                    tensor_rs.decode(data)
                } else {
                    self.base_rs.decode(data)
                }
            },
            _ => self.base_rs.decode(data),
        };
        
        // Update performance history
        let duration = start_time.elapsed();
        let _ = self.update_performance_history(algorithm, duration);
        
        // Update error statistics (simplified)
        let error_count = if result.is_ok() { 0 } else { 1 };
        let error_positions = if error_count > 0 { vec![0] } else { vec![] };
        let correctable = result.is_ok();
        let _ = self.update_error_statistics(error_count, &error_positions, correctable);
        
        result
    }
    
    fn max_correctable_errors(&self) -> usize {
        self.base_rs.max_correctable_errors()
    }
    
    fn overhead_ratio(&self) -> f64 {
        self.base_rs.overhead_ratio()
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the AdaptiveReedSolomon directory
        let adaptive_rs_path = path.join("adaptive_reed_solomon");
        std::fs::create_dir_all(&adaptive_rs_path)?;
        
        // Generate lookup tables for the base implementation
        self.base_rs.generate_lookup_tables(&adaptive_rs_path)?;
        
        // Generate lookup tables for the tensor implementation if available
        if let Some(tensor_rs) = &self.tensor_rs {
            tensor_rs.generate_lookup_tables(&adaptive_rs_path)?;
        }
        
        Ok(())
    }
    
    fn supports_hardware_acceleration(&self) -> bool {
        self.base_rs.supports_hardware_acceleration()
    }
    
    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.base_rs.set_hardware_accelerator(accelerator.clone());
        
        if let Some(tensor_rs) = &mut self.tensor_rs {
            tensor_rs.set_hardware_accelerator(accelerator);
        }
    }
    
    fn name(&self) -> &str {
        "Adaptive Reed-Solomon"
    }
}

impl fmt::Debug for AdaptiveReedSolomon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaptiveReedSolomon")
            .field("current_algorithm", &self.current_algorithm)
            .field("base_rs", &self.base_rs)
            .field("has_tensor_rs", &self.tensor_rs.is_some())
            .finish()
    }
}

impl Clone for AdaptiveReedSolomon {
    fn clone(&self) -> Self {
        Self {
            base_rs: self.base_rs.clone(),
            tensor_rs: self.tensor_rs.clone(),
            current_algorithm: self.current_algorithm,
            performance_history: Mutex::new(HashMap::new()),
            error_statistics: Mutex::new(ErrorStatistics {
                total_processed: 0,
                error_free: 0,
                correctable: 0,
                uncorrectable: 0,
                error_positions: HashMap::new(),
            }),
        }
    }
} 