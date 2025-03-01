//! Convolutional code implementation for error correction.
//!
//! Convolutional codes are a type of error-correcting code in which each m-bit information
//! symbol to be encoded is transformed into an n-bit symbol, where m/n is the code rate
//! and the transformation is a function of the last k information symbols, where k is the
//! constraint length of the code.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use super::{AlgorithmType, ErrorCorrectionAlgorithm};
use crate::error::{Error, Result};
use crate::hardware::HardwareAccelerator;

/// Implementation of Convolutional codes for error correction.
pub struct ConvolutionalCode {
    /// The constraint length (K)
    constraint_length: usize,
    /// The code rate (m/n)
    code_rate_numerator: usize,
    /// The code rate (m/n)
    code_rate_denominator: usize,
    /// The generator polynomials
    generator_polynomials: Vec<u64>,
    /// The trellis structure for Viterbi decoding
    trellis: Option<Trellis>,
    /// The hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
}

/// Trellis structure for Viterbi decoding
struct Trellis {
    /// Number of states in the trellis
    states: usize,
    /// State transitions
    next_states: Vec<Vec<usize>>,
    /// Output bits for each transition
    outputs: Vec<Vec<Vec<u8>>>,
}

impl ConvolutionalCode {
    /// Creates a new Convolutional code instance with default parameters.
    ///
    /// # Arguments
    ///
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `ConvolutionalCode` instance or an error if creation failed.
    pub fn new(hardware_accelerator: Arc<dyn HardwareAccelerator>) -> Result<Self> {
        // Default parameters for a rate 1/2, constraint length 7 convolutional code
        Self::with_params(7, 1, 2, vec![0o171, 0o133], hardware_accelerator)
    }

    /// Creates a new Convolutional code instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `constraint_length` - The constraint length (K)
    /// * `code_rate_numerator` - The code rate numerator (m)
    /// * `code_rate_denominator` - The code rate denominator (n)
    /// * `generator_polynomials` - The generator polynomials
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `ConvolutionalCode` instance or an error if parameters are invalid.
    pub fn with_params(
        constraint_length: usize,
        code_rate_numerator: usize,
        code_rate_denominator: usize,
        generator_polynomials: Vec<u64>,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        if constraint_length < 2 || constraint_length > 64 {
            return Err(Error::InvalidInput(
                "Constraint length must be between 2 and 64".into(),
            ));
        }

        if code_rate_numerator == 0 || code_rate_denominator == 0 {
            return Err(Error::InvalidInput(
                "Code rate numerator and denominator must be positive".into(),
            ));
        }

        if code_rate_numerator >= code_rate_denominator {
            return Err(Error::InvalidInput("Code rate must be less than 1".into()));
        }

        if generator_polynomials.len() != code_rate_denominator {
            return Err(Error::InvalidInput(
                format!(
                    "Number of generator polynomials ({}) must match code rate denominator ({})",
                    generator_polynomials.len(),
                    code_rate_denominator
                )
                .into(),
            ));
        }

        // Build trellis structure for Viterbi decoding
        let trellis = Self::build_trellis(constraint_length, &generator_polynomials);

        Ok(Self {
            constraint_length,
            code_rate_numerator,
            code_rate_denominator,
            generator_polynomials,
            trellis: Some(trellis),
            hardware_accelerator,
        })
    }

    /// Builds the trellis structure for Viterbi decoding.
    ///
    /// # Arguments
    ///
    /// * `constraint_length` - The constraint length (K)
    /// * `generator_polynomials` - The generator polynomials
    ///
    /// # Returns
    ///
    /// The trellis structure.
    fn build_trellis(constraint_length: usize, generator_polynomials: &[u64]) -> Trellis {
        let num_states = 1 << (constraint_length - 1);
        let num_inputs = 1 << 1; // Binary input

        let mut next_states = vec![vec![0; num_inputs]; num_states];
        let mut outputs = vec![vec![vec![0; generator_polynomials.len()]; num_inputs]; num_states];

        for state in 0..num_states {
            for input in 0..num_inputs {
                // Calculate next state
                let next_state = ((state << 1) | input) & (num_states - 1);
                next_states[state][input] = next_state;

                // Calculate output bits
                let register = (state << 1) | input;
                for (i, &poly) in generator_polynomials.iter().enumerate() {
                    let mut output_bit = 0;
                    for j in 0..constraint_length {
                        if (poly & (1 << j)) != 0 {
                            output_bit ^= (register >> j) & 1;
                        }
                    }
                    outputs[state][input][i] = output_bit as u8;
                }
            }
        }

        Trellis {
            states: num_states,
            next_states,
            outputs,
        }
    }

    /// Encodes a message using the convolutional encoder.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to encode
    ///
    /// # Returns
    ///
    /// The encoded codeword.
    fn encode_message(&self, message: &[u8]) -> Result<Vec<u8>> {
        // Convert message to bits
        let mut message_bits = Vec::with_capacity(message.len() * 8);
        for &byte in message {
            for i in 0..8 {
                message_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Encode bits
        let mut encoded_bits = Vec::with_capacity(
            message_bits.len() * self.code_rate_denominator / self.code_rate_numerator,
        );
        let mut shift_reg = 0u64;

        for &bit in &message_bits {
            // Shift in the new bit
            shift_reg = ((shift_reg << 1) | (bit as u64)) & ((1 << self.constraint_length) - 1);

            // Calculate output bits for each generator polynomial
            for &poly in &self.generator_polynomials {
                let mut output_bit = 0;
                for j in 0..self.constraint_length {
                    if (poly & (1 << j)) != 0 {
                        output_bit ^= (shift_reg >> j) & 1;
                    }
                }
                encoded_bits.push(output_bit as u8);
            }
        }

        // Add termination bits (flush the shift register)
        for _ in 0..(self.constraint_length - 1) {
            shift_reg = (shift_reg << 1) & ((1 << self.constraint_length) - 1);

            for &poly in &self.generator_polynomials {
                let mut output_bit = 0;
                for j in 0..self.constraint_length {
                    if (poly & (1 << j)) != 0 {
                        output_bit ^= (shift_reg >> j) & 1;
                    }
                }
                encoded_bits.push(output_bit as u8);
            }
        }

        // Convert bits back to bytes
        let mut encoded = Vec::with_capacity((encoded_bits.len() + 7) / 8);
        for chunk in encoded_bits.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if i < 8 {
                    byte |= bit << (7 - i);
                }
            }
            encoded.push(byte);
        }

        Ok(encoded)
    }

    /// Decodes a codeword using the Viterbi algorithm.
    ///
    /// # Arguments
    ///
    /// * `codeword` - The codeword to decode
    ///
    /// # Returns
    ///
    /// The decoded message.
    fn decode_codeword(&self, codeword: &[u8]) -> Result<Vec<u8>> {
        // Convert codeword to bits
        let mut codeword_bits = Vec::with_capacity(codeword.len() * 8);
        for &byte in codeword {
            for i in 0..8 {
                codeword_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Check if we have a valid trellis
        let trellis = match &self.trellis {
            Some(t) => t,
            None => return Err(Error::Internal("Trellis not initialized".to_string())),
        };

        // Group bits into symbols
        let mut symbols = Vec::with_capacity(codeword_bits.len() / self.code_rate_denominator);
        for chunk in codeword_bits.chunks(self.code_rate_denominator) {
            if chunk.len() == self.code_rate_denominator {
                symbols.push(chunk.to_vec());
            }
        }

        // Initialize path metrics and traceback
        let num_states = trellis.states;
        let mut path_metrics = vec![f64::INFINITY; num_states];
        path_metrics[0] = 0.0; // Start at state 0

        let mut traceback = vec![vec![0; num_states]; symbols.len()];

        // Forward pass (Viterbi algorithm)
        for (t, symbol) in symbols.iter().enumerate() {
            let mut new_path_metrics = vec![f64::INFINITY; num_states];

            for state in 0..num_states {
                for input in 0..2 {
                    let next_state = trellis.next_states[state][input];
                    let output = &trellis.outputs[state][input];

                    // Calculate Hamming distance between output and received symbol
                    let mut distance = 0.0;
                    for i in 0..output.len() {
                        if i < symbol.len() && output[i] != symbol[i] {
                            distance += 1.0;
                        }
                    }

                    let metric = path_metrics[state] + distance;
                    if metric < new_path_metrics[next_state] {
                        new_path_metrics[next_state] = metric;
                        traceback[t][next_state] = state;
                    }
                }
            }

            path_metrics = new_path_metrics;
        }

        // Find the state with the best metric
        let mut best_state = 0;
        let mut best_metric = path_metrics[0];
        for state in 1..num_states {
            if path_metrics[state] < best_metric {
                best_metric = path_metrics[state];
                best_state = state;
            }
        }

        // Traceback to find the input sequence
        let mut decoded_bits = Vec::with_capacity(symbols.len());
        let mut state = best_state;

        for t in (0..symbols.len()).rev() {
            let prev_state = traceback[t][state];
            let input = if trellis.next_states[prev_state][1] == state {
                1
            } else {
                0
            };
            decoded_bits.push(input as u8);
            state = prev_state;
        }

        // Reverse the bits (traceback goes backwards)
        decoded_bits.reverse();

        // Remove termination bits
        if decoded_bits.len() > self.constraint_length - 1 {
            decoded_bits.truncate(decoded_bits.len() - (self.constraint_length - 1));
        }

        // Convert bits back to bytes
        let mut decoded = Vec::with_capacity((decoded_bits.len() + 7) / 8);
        for chunk in decoded_bits.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if i < 8 {
                    byte |= bit << (7 - i);
                }
            }
            decoded.push(byte);
        }

        Ok(decoded)
    }

    /// Punctures a codeword to achieve a higher code rate.
    ///
    /// # Arguments
    ///
    /// * `codeword` - The codeword to puncture
    /// * `pattern` - The puncturing pattern (1 = keep, 0 = discard)
    ///
    /// # Returns
    ///
    /// The punctured codeword.
    pub fn puncture(&self, codeword: &[u8], pattern: &[u8]) -> Result<Vec<u8>> {
        if pattern.is_empty() {
            return Err(Error::InvalidInput(
                "Puncturing pattern cannot be empty".into(),
            ));
        }

        // Convert codeword to bits
        let mut codeword_bits = Vec::with_capacity(codeword.len() * 8);
        for &byte in codeword {
            for i in 0..8 {
                codeword_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Apply puncturing pattern
        let mut punctured_bits = Vec::with_capacity(codeword_bits.len());
        for (i, &bit) in codeword_bits.iter().enumerate() {
            if pattern[i % pattern.len()] != 0 {
                punctured_bits.push(bit);
            }
        }

        // Convert bits back to bytes
        let mut punctured = Vec::with_capacity((punctured_bits.len() + 7) / 8);
        for chunk in punctured_bits.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if i < 8 {
                    byte |= bit << (7 - i);
                }
            }
            punctured.push(byte);
        }

        Ok(punctured)
    }

    /// Depunctures a codeword by inserting erasures at punctured positions.
    ///
    /// # Arguments
    ///
    /// * `punctured` - The punctured codeword
    /// * `pattern` - The puncturing pattern (1 = keep, 0 = discard)
    /// * `erasure_value` - The value to insert at punctured positions
    ///
    /// # Returns
    ///
    /// The depunctured codeword.
    pub fn depuncture(
        &self,
        punctured: &[u8],
        pattern: &[u8],
        erasure_value: u8,
    ) -> Result<Vec<u8>> {
        if pattern.is_empty() {
            return Err(Error::InvalidInput(
                "Puncturing pattern cannot be empty".into(),
            ));
        }

        // Convert punctured codeword to bits
        let mut punctured_bits = Vec::with_capacity(punctured.len() * 8);
        for &byte in punctured {
            for i in 0..8 {
                punctured_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Calculate original size
        let ones_in_pattern = pattern.iter().filter(|&&b| b != 0).count();
        let original_size = punctured_bits.len() * pattern.len() / ones_in_pattern;

        // Apply depuncturing
        let mut depunctured_bits = Vec::with_capacity(original_size);
        let mut punctured_idx = 0;

        for i in 0..original_size {
            if pattern[i % pattern.len()] != 0 {
                if punctured_idx < punctured_bits.len() {
                    depunctured_bits.push(punctured_bits[punctured_idx]);
                    punctured_idx += 1;
                } else {
                    depunctured_bits.push(erasure_value);
                }
            } else {
                depunctured_bits.push(erasure_value);
            }
        }

        // Convert bits back to bytes
        let mut depunctured = Vec::with_capacity((depunctured_bits.len() + 7) / 8);
        for chunk in depunctured_bits.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if i < 8 {
                    byte |= bit << (7 - i);
                }
            }
            depunctured.push(byte);
        }

        Ok(depunctured)
    }
}

impl ErrorCorrectionAlgorithm for ConvolutionalCode {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Convolutional
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available()
            && self.hardware_accelerator.supports_convolutional()
        {
            return self
                .hardware_accelerator
                .convolutional_encode(data, self.constraint_length, &self.generator_polynomials)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Software implementation
        self.encode_message(data)
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available()
            && self.hardware_accelerator.supports_convolutional()
        {
            return self
                .hardware_accelerator
                .convolutional_decode(data, self.constraint_length, &self.generator_polynomials)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Software implementation
        self.decode_codeword(data)
    }

    fn max_correctable_errors(&self) -> usize {
        // For a rate 1/n convolutional code with hard decision decoding,
        // the free distance is approximately (n+1)*K/2 for large K
        // and the code can correct up to (free_distance-1)/2 errors
        let free_distance = (self.code_rate_denominator + 1) * self.constraint_length / 2;
        (free_distance - 1) / 2
    }

    fn overhead_ratio(&self) -> f64 {
        self.code_rate_denominator as f64 / self.code_rate_numerator as f64
    }

    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the Convolutional directory
        let conv_path = path.join("convolutional");
        std::fs::create_dir_all(&conv_path)?;

        // Save trellis structure
        if let Some(trellis) = &self.trellis {
            // Save next states
            let next_states_path = conv_path.join("next_states.bin");
            let next_states_data = bincode::serialize(&trellis.next_states).map_err(|_| {
                Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                    "Failed to serialize next states".to_string(),
                )))
            })?;

            std::fs::write(next_states_path, next_states_data)?;

            // Save outputs
            let outputs_path = conv_path.join("outputs.bin");
            let outputs_data = bincode::serialize(&trellis.outputs).map_err(|_| {
                Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                    "Failed to serialize outputs".to_string(),
                )))
            })?;

            std::fs::write(outputs_path, outputs_data)?;
        }

        // Save generator polynomials
        let poly_path = conv_path.join("generator_polynomials.bin");
        let poly_data = bincode::serialize(&self.generator_polynomials).map_err(|_| {
            Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                "Failed to serialize generator polynomials".to_string(),
            )))
        })?;

        std::fs::write(poly_path, poly_data)?;

        Ok(())
    }

    fn supports_hardware_acceleration(&self) -> bool {
        self.hardware_accelerator.is_available()
            && self.hardware_accelerator.supports_convolutional()
    }

    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator;
    }

    fn name(&self) -> &str {
        "Convolutional Code"
    }
}

impl fmt::Debug for ConvolutionalCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConvolutionalCode")
            .field("constraint_length", &self.constraint_length)
            .field(
                "code_rate",
                &format!(
                    "{}/{}",
                    self.code_rate_numerator, self.code_rate_denominator
                ),
            )
            .field("generator_polynomials", &self.generator_polynomials)
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .finish()
    }
}
