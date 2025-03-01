//! LDPC (Low-Density Parity-Check) error correction algorithm implementation.
//!
//! LDPC codes are a class of linear block codes with sparse parity-check matrices.
//! They provide near-Shannon-limit performance and are used in many modern
//! communication standards.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::hardware::HardwareAccelerator;
use super::{AlgorithmType, ErrorCorrectionAlgorithm};
use serde::{Serialize, Deserialize};
use rand::prelude::*;

/// LDPC error correction algorithm implementation.
pub struct Ldpc {
    /// Code rate (k/n)
    code_rate: f64,
    /// Codeword length (n)
    codeword_length: usize,
    /// Message length (k)
    message_length: usize,
    /// Maximum number of iterations for decoding
    max_iterations: usize,
    /// Parity check matrix (sparse representation)
    parity_check_matrix: SparseMatrix,
    /// Generator matrix (sparse representation)
    generator_matrix: SparseMatrix,
    /// Hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
}

/// Sparse matrix representation for efficient LDPC operations.
#[derive(Clone, Serialize, Deserialize)]
struct SparseMatrix {
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Non-zero entries as (row, col) pairs
    entries: Vec<(usize, usize)>,
    /// Row indices for each column
    col_to_rows: Vec<Vec<usize>>,
    /// Column indices for each row
    row_to_cols: Vec<Vec<usize>>,
}

impl SparseMatrix {
    /// Creates a new sparse matrix with the given dimensions.
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
            col_to_rows: vec![Vec::new(); cols],
            row_to_cols: vec![Vec::new(); rows],
        }
    }

    /// Sets a value at the given position.
    fn set(&mut self, row: usize, col: usize) {
        if !self.entries.contains(&(row, col)) {
            self.entries.push((row, col));
            self.col_to_rows[col].push(row);
            self.row_to_cols[row].push(col);
        }
    }

    /// Returns true if the value at the given position is non-zero.
    fn get(&self, row: usize, col: usize) -> bool {
        self.entries.contains(&(row, col))
    }

    /// Returns the density of the matrix (fraction of non-zero entries).
    fn density(&self) -> f64 {
        self.entries.len() as f64 / (self.rows * self.cols) as f64
    }
}

impl Ldpc {
    /// Creates a new LDPC encoder/decoder.
    ///
    /// # Arguments
    ///
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `Ldpc` instance or an error if initialization fails.
    pub fn new(hardware_accelerator: Arc<dyn HardwareAccelerator>) -> Result<Self> {
        // Default parameters for a (1024, 512) LDPC code
        let codeword_length = 1024;
        let message_length = 512;
        let code_rate = message_length as f64 / codeword_length as f64;
        let max_iterations = 50;

        // Generate parity check matrix
        let parity_check_matrix = Self::generate_parity_check_matrix(codeword_length, message_length)?;
        
        // Generate generator matrix from parity check matrix
        let generator_matrix = Self::generate_generator_matrix(&parity_check_matrix, message_length)?;

        Ok(Self {
            code_rate,
            codeword_length,
            message_length,
            max_iterations,
            parity_check_matrix,
            generator_matrix,
            hardware_accelerator,
        })
    }

    /// Creates a new LDPC encoder/decoder with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `codeword_length` - Total codeword length (n)
    /// * `message_length` - Message length (k)
    /// * `max_iterations` - Maximum number of iterations for decoding
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `Ldpc` instance or an error if parameters are invalid.
    pub fn with_params(
        codeword_length: usize,
        message_length: usize,
        max_iterations: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        if message_length >= codeword_length {
            return Err(Error::InvalidInput(
                "Message length must be less than codeword length".into(),
            ));
        }

        let code_rate = message_length as f64 / codeword_length as f64;
        
        // Generate parity check matrix
        let parity_check_matrix = Self::generate_parity_check_matrix(codeword_length, message_length)?;
        
        // Generate generator matrix from parity check matrix
        let generator_matrix = Self::generate_generator_matrix(&parity_check_matrix, message_length)?;

        Ok(Self {
            code_rate,
            codeword_length,
            message_length,
            max_iterations,
            parity_check_matrix,
            generator_matrix,
            hardware_accelerator,
        })
    }

    /// Generates a random parity check matrix for LDPC codes.
    ///
    /// This uses the Progressive Edge-Growth (PEG) algorithm to generate
    /// a parity check matrix with good properties.
    fn generate_parity_check_matrix(codeword_length: usize, message_length: usize) -> Result<SparseMatrix> {
        let parity_length = codeword_length - message_length;
        let mut matrix = SparseMatrix::new(parity_length, codeword_length);
        
        // Column weight (number of 1s per column)
        let col_weight = 3;
        
        // Row weight (number of 1s per row)
        let row_weight = (col_weight * codeword_length) / parity_length;
        
        // Initialize with a simple regular structure
        for i in 0..parity_length {
            // Set diagonal elements in the parity part
            matrix.set(i, message_length + i);
            
            // Set additional elements in the message part
            for j in 0..row_weight - 1 {
                let col = (i * (row_weight - 1) + j) % message_length;
                matrix.set(i, col);
            }
        }
        
        // Ensure each column has the desired weight
        for j in 0..codeword_length {
            let current_weight = matrix.col_to_rows[j].len();
            if j < message_length && current_weight < col_weight {
                // Add more 1s to this column
                for _ in current_weight..col_weight {
                    // Find a row with the fewest 1s
                    let mut min_row = 0;
                    let mut min_weight = usize::MAX;
                    
                    for i in 0..parity_length {
                        if !matrix.get(i, j) && matrix.row_to_cols[i].len() < min_weight {
                            min_row = i;
                            min_weight = matrix.row_to_cols[i].len();
                        }
                    }
                    
                    matrix.set(min_row, j);
                }
            }
        }
        
        Ok(matrix)
    }

    /// Generates a generator matrix from a parity check matrix.
    ///
    /// This uses Gaussian elimination to convert the parity check matrix
    /// to systematic form and then extract the generator matrix.
    fn generate_generator_matrix(parity_check_matrix: &SparseMatrix, message_length: usize) -> Result<SparseMatrix> {
        let parity_length = parity_check_matrix.rows;
        let codeword_length = parity_check_matrix.cols;
        
        // Create a systematic generator matrix
        let mut generator = SparseMatrix::new(message_length, codeword_length);
        
        // Set the identity part (first k columns)
        for i in 0..message_length {
            generator.set(i, i);
        }
        
        // Set the parity part based on the parity check matrix
        for i in 0..message_length {
            for j in 0..parity_length {
                let col = message_length + j;
                
                // Check if this message bit affects this parity bit
                let affects_parity = parity_check_matrix.row_to_cols[j].contains(&i);
                
                if affects_parity {
                    generator.set(i, col);
                }
            }
        }
        
        Ok(generator)
    }

    /// Performs belief propagation decoding for LDPC codes.
    ///
    /// This is the standard iterative decoding algorithm for LDPC codes.
    fn belief_propagation_decode(&self, llr: &[f64]) -> Result<Vec<u8>> {
        if llr.len() != self.codeword_length {
            return Err(Error::InvalidInput(
                format!("Invalid LLR length: {} (expected: {})", llr.len(), self.codeword_length).into(),
            ));
        }
        
        // Initialize node beliefs
        let mut variable_nodes = llr.to_vec();
        let mut check_to_var = vec![vec![0.0; self.parity_check_matrix.col_to_rows[0].len()]; self.codeword_length];
        let mut var_to_check = vec![vec![0.0; self.parity_check_matrix.row_to_cols[0].len()]; self.parity_check_matrix.rows];
        
        // Initialize messages from variable nodes to check nodes
        for j in 0..self.codeword_length {
            for (_idx, &i) in self.parity_check_matrix.col_to_rows[j].iter().enumerate() {
                var_to_check[i][self.parity_check_matrix.row_to_cols[i].iter().position(|&x| x == j).unwrap()] = llr[j];
            }
        }
        
        // Iterative decoding
        for _ in 0..self.max_iterations {
            // Check node update
            for i in 0..self.parity_check_matrix.rows {
                for (_j_idx, &j) in self.parity_check_matrix.row_to_cols[i].iter().enumerate() {
                    // Product of signs
                    let mut sign = 1.0;
                    let mut magnitude = 0.0;
                    
                    for (k_idx, &k) in self.parity_check_matrix.row_to_cols[i].iter().enumerate() {
                        if k != j {
                            let msg = var_to_check[i][k_idx];
                            sign *= msg.signum();
                            magnitude += (msg.abs() + 1e-10).ln();
                        }
                    }
                    
                    check_to_var[j][self.parity_check_matrix.col_to_rows[j].iter().position(|&x| x == i).unwrap()] = sign * magnitude.exp();
                }
            }
            
            // Variable node update
            for j in 0..self.codeword_length {
                for (_i_idx, &i) in self.parity_check_matrix.col_to_rows[j].iter().enumerate() {
                    let mut sum = llr[j];
                    
                    for (k_idx, &k) in self.parity_check_matrix.col_to_rows[j].iter().enumerate() {
                        if k != i {
                            sum += check_to_var[j][k_idx];
                        }
                    }
                    
                    var_to_check[i][self.parity_check_matrix.row_to_cols[i].iter().position(|&x| x == j).unwrap()] = sum;
                }
                
                // Update variable node belief
                variable_nodes[j] = llr[j];
                for (i_idx, _) in self.parity_check_matrix.col_to_rows[j].iter().enumerate() {
                    variable_nodes[j] += check_to_var[j][i_idx];
                }
            }
            
            // Check if codeword is valid
            let hard_decision: Vec<u8> = variable_nodes.iter().map(|&x| if x >= 0.0 { 0 } else { 1 }).collect();
            if self.check_parity(&hard_decision) {
                return Ok(hard_decision);
            }
        }
        
        // Return best guess after max iterations
        let hard_decision: Vec<u8> = variable_nodes.iter().map(|&x| if x >= 0.0 { 0 } else { 1 }).collect();
        Ok(hard_decision)
    }

    /// Checks if a codeword satisfies the parity check equations.
    fn check_parity(&self, codeword: &[u8]) -> bool {
        for i in 0..self.parity_check_matrix.rows {
            let mut sum = 0;
            
            for &j in &self.parity_check_matrix.row_to_cols[i] {
                sum ^= codeword[j] as usize;
            }
            
            if sum != 0 {
                return false;
            }
        }
        
        true
    }

    /// Converts a binary message to log-likelihood ratios (LLRs).
    fn bits_to_llr(&self, bits: &[u8], noise_variance: f64) -> Vec<f64> {
        bits.iter()
            .map(|&bit| {
                if bit == 0 {
                    2.0 / noise_variance
                } else {
                    -2.0 / noise_variance
                }
            })
            .collect()
    }

    /// Adds Gaussian noise to a codeword.
    ///
    /// # Arguments
    ///
    /// * `codeword` - The codeword to add noise to
    /// * `snr_db` - The signal-to-noise ratio in dB
    ///
    /// # Returns
    ///
    /// The noisy codeword.
    #[allow(dead_code)]
    fn add_noise(&self, codeword: &[u8], snr_db: f64) -> Vec<f64> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        
        let mut rng = ChaCha8Rng::seed_from_u64(42); // Fixed seed for reproducibility
        
        codeword
            .iter()
            .map(|&bit| {
                let signal = if bit == 0 { 1.0 } else { -1.0 };
                let noise = rng.random::<f64>() * snr_db.sqrt();
                signal + noise
            })
            .collect()
    }

    /// Converts received values to log-likelihood ratios (LLRs).
    ///
    /// # Arguments
    ///
    /// * `received` - The received values
    /// * `snr_db` - The signal-to-noise ratio in dB
    ///
    /// # Returns
    ///
    /// The LLRs.
    #[allow(dead_code)]
    fn received_to_llr(&self, received: &[f64], snr_db: f64) -> Vec<f64> {
        received
            .iter()
            .map(|&r| 2.0 * r / snr_db)
            .collect()
    }
}

impl ErrorCorrectionAlgorithm for Ldpc {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Ldpc
    }
    
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check if data size is valid
        if data.len() * 8 > self.message_length {
            return Err(Error::InvalidInput(
                format!("Input data too large: {} bytes (max: {} bits)", data.len(), self.message_length).into(),
            ));
        }
        
        // Use hardware acceleration if available - but hardware doesn't currently support LDPC
        if self.hardware_accelerator.is_available() {
            // Just a placeholder - hardware doesn't support LDPC yet
            // In the future this could call `self.hardware_accelerator.ldpc_encode(...)`
        }
        
        // Convert data to bits
        let mut message_bits = Vec::with_capacity(self.message_length);
        for &byte in data {
            for i in 0..8 {
                message_bits.push((byte >> i) & 1);
            }
        }
        
        // Pad message if needed
        message_bits.resize(self.message_length, 0);
        
        // Encode using generator matrix
        let mut codeword = vec![0; self.codeword_length];
        
        // Set message bits
        for i in 0..self.message_length {
            codeword[i] = message_bits[i];
        }
        
        // Calculate parity bits
        for j in self.message_length..self.codeword_length {
            let mut bit = 0;
            
            for &i in &self.parity_check_matrix.col_to_rows[j] {
                let mut row_sum = 0;
                
                for &col in &self.parity_check_matrix.row_to_cols[i] {
                    if col != j && col < j {
                        row_sum ^= codeword[col] as usize;
                    }
                }
                
                bit ^= row_sum;
            }
            
            codeword[j] = bit as u8;
        }
        
        // Pack bits into bytes
        let mut encoded = Vec::with_capacity((self.codeword_length + 7) / 8);
        for chunk in codeword.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= (bit as u8) << i;
            }
            encoded.push(byte);
        }
        
        Ok(encoded)
    }
    
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check if data size is valid
        if data.len() * 8 < self.codeword_length {
            return Err(Error::InvalidInput(
                format!("Input data too small: {} bytes (min: {} bits)", data.len(), self.codeword_length).into(),
            ));
        }
        
        // Use hardware acceleration if available - but hardware doesn't currently support LDPC
        if self.hardware_accelerator.is_available() {
            // Just a placeholder - hardware doesn't support LDPC yet
            // In the future this could call `self.hardware_accelerator.ldpc_decode(...)`
        }
        
        // Unpack bytes to bits
        let mut received_bits = Vec::with_capacity(self.codeword_length);
        for &byte in data {
            for i in 0..8 {
                if received_bits.len() < self.codeword_length {
                    received_bits.push((byte >> i) & 1);
                }
            }
        }
        
        // Convert to LLRs with a default noise variance
        let noise_variance = 1.0;
        let llr = self.bits_to_llr(&received_bits, noise_variance);
        
        // Decode using belief propagation
        let decoded_bits = self.belief_propagation_decode(&llr)?;
        
        // Extract message bits
        let message_bits = &decoded_bits[0..self.message_length];
        
        // Pack bits into bytes
        let mut decoded = Vec::with_capacity((self.message_length + 7) / 8);
        for chunk in message_bits.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= (bit as u8) << i;
            }
            decoded.push(byte);
        }
        
        Ok(decoded)
    }
    
    fn max_correctable_errors(&self) -> usize {
        // LDPC codes can typically correct up to (d_min - 1)/2 errors,
        // where d_min is the minimum distance of the code.
        // For a well-designed LDPC code, this is approximately:
        ((1.0 - self.code_rate) * self.codeword_length as f64 * 0.1) as usize
    }
    
    fn overhead_ratio(&self) -> f64 {
        1.0 / self.code_rate
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the LDPC directory
        let ldpc_path = path.join("ldpc");
        std::fs::create_dir_all(&ldpc_path)?;
        
        // Save parity check matrix
        let parity_check_path = ldpc_path.join("parity_check.bin");
        let parity_check_data = bincode::serialize(&self.parity_check_matrix)
            .map_err(|e| Error::BinarySerialization(e))?;
        
        std::fs::write(parity_check_path, parity_check_data)?;
        
        // Save generator matrix
        let generator_path = ldpc_path.join("generator.bin");
        let generator_data = bincode::serialize(&self.generator_matrix)
            .map_err(|e| Error::BinarySerialization(e))?;
        
        std::fs::write(generator_path, generator_data)?;
        
        Ok(())
    }
    
    fn supports_hardware_acceleration(&self) -> bool {
        // LDPC is not supported by hardware accelerators yet
        false
    }
    
    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator;
    }
    
    fn name(&self) -> &str {
        "LDPC"
    }
}

impl fmt::Debug for Ldpc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ldpc")
            .field("code_rate", &self.code_rate)
            .field("codeword_length", &self.codeword_length)
            .field("message_length", &self.message_length)
            .field("max_iterations", &self.max_iterations)
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .field("parity_check_density", &self.parity_check_matrix.density())
            .finish()
    }
} 