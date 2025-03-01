//! Reed-Muller error correction algorithm implementation.
//!
//! Reed-Muller codes are a family of linear error-correcting codes that can
//! correct multiple errors. They are particularly useful for applications
//! requiring moderate error correction capabilities.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::galois::GaloisField;
use crate::hardware::HardwareAccelerator;
use super::{AlgorithmType, ErrorCorrectionAlgorithm};

/// Reed-Muller error correction algorithm implementation.
pub struct ReedMullerCode {
    /// Galois field used for finite field arithmetic
    galois_field: Arc<GaloisField>,
    /// Order of the Reed-Muller code (r)
    order: usize,
    /// Number of variables (m)
    num_variables: usize,
    /// Total codeword length (2^m)
    codeword_length: usize,
    /// Message length (sum of binomial coefficients)
    message_length: usize,
    /// Generator matrix
    generator_matrix: Vec<Vec<u16>>,
    /// Hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
}

impl ReedMullerCode {
    /// Creates a new Reed-Muller encoder/decoder.
    ///
    /// # Arguments
    ///
    /// * `galois_field` - Galois field for finite field arithmetic
    /// * `order` - Order of the Reed-Muller code (r)
    /// * `num_variables` - Number of variables (m)
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `ReedMullerCode` instance or an error if parameters are invalid.
    pub fn new(
        galois_field: Arc<GaloisField>,
        order: usize,
        num_variables: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        // Validate parameters
        if order >= num_variables {
            return Err(Error::InvalidInput(
                "Order must be less than the number of variables".into(),
            ));
        }

        // Calculate codeword length and message length
        let codeword_length = 1 << num_variables; // 2^m
        let message_length = Self::calculate_message_length(order, num_variables);

        // Generate the generator matrix
        let generator_matrix = Self::generate_generator_matrix(order, num_variables)?;

        Ok(Self {
            galois_field,
            order,
            num_variables,
            codeword_length,
            message_length,
            generator_matrix,
            hardware_accelerator,
        })
    }

    /// Calculates the message length for a Reed-Muller code.
    ///
    /// # Arguments
    ///
    /// * `order` - Order of the Reed-Muller code (r)
    /// * `num_variables` - Number of variables (m)
    ///
    /// # Returns
    ///
    /// The message length.
    fn calculate_message_length(order: usize, num_variables: usize) -> usize {
        let mut length = 0;
        for i in 0..=order {
            length += Self::binomial(num_variables, i);
        }
        length
    }

    /// Calculates the binomial coefficient (n choose k).
    ///
    /// # Arguments
    ///
    /// * `n` - The total number of items
    /// * `k` - The number of items to choose
    ///
    /// # Returns
    ///
    /// The binomial coefficient.
    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        
        let k = k.min(n - k);
        let mut c = 1;
        for i in 0..k {
            c = c * (n - i) / (i + 1);
        }
        c
    }

    /// Generates the generator matrix for a Reed-Muller code.
    ///
    /// # Arguments
    ///
    /// * `order` - Order of the Reed-Muller code (r)
    /// * `num_variables` - Number of variables (m)
    ///
    /// # Returns
    ///
    /// The generator matrix.
    fn generate_generator_matrix(order: usize, num_variables: usize) -> Result<Vec<Vec<u16>>> {
        let codeword_length = 1 << num_variables;
        let message_length = Self::calculate_message_length(order, num_variables);
        
        let mut generator_matrix = vec![vec![0; codeword_length]; message_length];
        
        // First row is all ones (constant term)
        for j in 0..codeword_length {
            generator_matrix[0][j] = 1;
        }
        
        // Generate rows for each monomial
        let mut row_index = 1;
        
        // For each order from 1 to r
        for r in 1..=order {
            // Generate all monomials of degree r
            let monomials = Self::generate_monomials(num_variables, r);
            
            for monomial in monomials {
                // For each possible input (represented as a binary number)
                for j in 0..codeword_length {
                    let mut product = 1;
                    
                    // Evaluate the monomial at this input
                    for &var in &monomial {
                        // Check if the var-th bit of j is set
                        if (j & (1 << var)) != 0 {
                            product &= 1;
                        } else {
                            product = 0;
                            break;
                        }
                    }
                    
                    generator_matrix[row_index][j] = product;
                }
                
                row_index += 1;
            }
        }
        
        Ok(generator_matrix)
    }

    /// Generates all monomials of a given degree.
    ///
    /// # Arguments
    ///
    /// * `num_variables` - Number of variables
    /// * `degree` - Degree of the monomials
    ///
    /// # Returns
    ///
    /// A vector of monomials, where each monomial is represented as a vector of variable indices.
    fn generate_monomials(num_variables: usize, degree: usize) -> Vec<Vec<usize>> {
        if degree == 0 {
            return vec![vec![]];
        }
        
        if degree == 1 {
            return (0..num_variables).map(|i| vec![i]).collect();
        }
        
        let mut result = Vec::new();
        
        // Generate all monomials of degree degree-1
        let lower_monomials = Self::generate_monomials(num_variables, degree - 1);
        
        for monomial in lower_monomials {
            let last_var = if monomial.is_empty() { 0 } else { monomial[monomial.len() - 1] };
            
            // Multiply by each variable with index >= last_var
            for var in last_var..num_variables {
                let mut new_monomial = monomial.clone();
                new_monomial.push(var);
                result.push(new_monomial);
            }
        }
        
        result
    }

    /// Encodes a message using the generator matrix.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to encode
    ///
    /// # Returns
    ///
    /// The encoded codeword.
    fn encode_with_generator_matrix(&self, message: &[u16]) -> Result<Vec<u16>> {
        if message.len() != self.message_length {
            return Err(Error::InvalidInput(
                format!("Invalid message length: {} (expected: {})", message.len(), self.message_length).into(),
            ));
        }
        
        let mut codeword = vec![0; self.codeword_length];
        
        for i in 0..self.message_length {
            if message[i] == 0 {
                continue;
            }
            
            for j in 0..self.codeword_length {
                codeword[j] = self.galois_field.add(codeword[j], self.galois_field.multiply(message[i], self.generator_matrix[i][j]));
            }
        }
        
        Ok(codeword)
    }

    /// Decodes a codeword using majority-logic decoding.
    ///
    /// # Arguments
    ///
    /// * `received` - The received codeword
    ///
    /// # Returns
    ///
    /// The decoded message.
    fn decode_with_majority_logic(&self, received: &[u16]) -> Result<Vec<u16>> {
        if received.len() != self.codeword_length {
            return Err(Error::InvalidInput(
                format!("Invalid codeword length: {} (expected: {})", received.len(), self.codeword_length).into(),
            ));
        }
        
        // For simplicity, we'll use a basic approach for first-order Reed-Muller codes
        // In a real implementation, this would use more sophisticated algorithms for higher orders
        
        let mut decoded = vec![0; self.message_length];
        
        // Decode the constant term (first bit of the message)
        let mut sum = 0;
        for &bit in received {
            sum += bit as usize;
        }
        decoded[0] = if sum > self.codeword_length / 2 { 1 } else { 0 };
        
        // For first-order terms, use the Hadamard transform
        if self.order >= 1 {
            for i in 0..self.num_variables {
                let mut sum = 0;
                
                for j in 0..self.codeword_length {
                    if (j & (1 << i)) != 0 {
                        sum += if received[j] == 1 { 1 } else { -1 };
                    } else {
                        sum += if received[j] == 1 { -1 } else { 1 };
                    }
                }
                
                decoded[i + 1] = if sum > 0 { 1 } else { 0 };
            }
        }
        
        // Higher-order terms would require more complex decoding
        
        Ok(decoded)
    }
}

impl ErrorCorrectionAlgorithm for ReedMullerCode {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::ReedMullerCode
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
        if self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_reed_muller() {
            return self.hardware_accelerator
                .reed_muller_encode(data, self.order, self.num_variables)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }
        
        // Software implementation
        let codeword = self.encode_with_generator_matrix(&message)?;
        
        // Convert back to bytes
        let encoded: Vec<u8> = codeword.iter().map(|&x| x as u8).collect();
        Ok(encoded)
    }
    
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() != self.codeword_length {
            return Err(Error::InvalidInput(
                format!("Invalid codeword length: {} (expected: {})", data.len(), self.codeword_length).into(),
            ));
        }
        
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_reed_muller() {
            return self.hardware_accelerator
                .reed_muller_decode(data, self.order, self.num_variables)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }
        
        // Convert data to field elements
        let received: Vec<u16> = data.iter().map(|&b| b as u16).collect();
        
        // Decode using majority-logic decoding
        let decoded = self.decode_with_majority_logic(&received)?;
        
        // Convert back to bytes
        let decoded_bytes: Vec<u8> = decoded.iter().map(|&x| x as u8).collect();
        Ok(decoded_bytes)
    }
    
    fn max_correctable_errors(&self) -> usize {
        // Reed-Muller codes of order r in m variables can correct up to (2^(m-r-1) - 1) errors
        (1 << (self.num_variables - self.order - 1)) - 1
    }
    
    fn overhead_ratio(&self) -> f64 {
        self.codeword_length as f64 / self.message_length as f64
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the Reed-Muller directory
        let rm_path = path.join("reed_muller");
        std::fs::create_dir_all(&rm_path)?;
        
        // Save generator matrix
        let gen_matrix_path = rm_path.join("generator_matrix.bin");
        let gen_matrix_data = bincode::serialize(&self.generator_matrix)
            .map_err(|e| Error::BinarySerialization(e))?;
        
        std::fs::write(gen_matrix_path, gen_matrix_data)?;
        
        Ok(())
    }
    
    fn supports_hardware_acceleration(&self) -> bool {
        self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_reed_muller()
    }
    
    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator;
    }
    
    fn name(&self) -> &str {
        "Reed-Muller Code"
    }
}

impl fmt::Debug for ReedMullerCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReedMullerCode")
            .field("order", &self.order)
            .field("num_variables", &self.num_variables)
            .field("codeword_length", &self.codeword_length)
            .field("message_length", &self.message_length)
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .finish()
    }
}

impl Clone for ReedMullerCode {
    fn clone(&self) -> Self {
        Self {
            galois_field: self.galois_field.clone(),
            order: self.order,
            num_variables: self.num_variables,
            codeword_length: self.codeword_length,
            message_length: self.message_length,
            generator_matrix: self.generator_matrix.clone(),
            hardware_accelerator: self.hardware_accelerator.clone(),
        }
    }
} 