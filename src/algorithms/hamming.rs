//! Hamming code implementation for error correction.
//!
//! Hamming codes are a family of linear error-correcting codes that can detect
//! up to two-bit errors or correct one-bit errors. They are widely used in
//! computer memory (RAM) and data transmission.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use super::{AlgorithmType, ErrorCorrectionAlgorithm};
use crate::error::{Error, Result};
use crate::hardware::HardwareAccelerator;

/// Implementation of Hamming codes for error correction.
pub struct HammingCode {
    /// The number of parity bits (r)
    parity_bits: usize,
    /// The total codeword length (2^r - 1)
    codeword_length: usize,
    /// The message length (2^r - r - 1)
    message_length: usize,
    /// The hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
}

impl HammingCode {
    /// Creates a new Hamming code instance.
    ///
    /// # Arguments
    ///
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `HammingCode` instance or an error if creation failed.
    pub fn new(hardware_accelerator: Arc<dyn HardwareAccelerator>) -> Result<Self> {
        // Default to (7,4) Hamming code
        Self::with_params(3, hardware_accelerator)
    }

    /// Creates a new Hamming code instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `parity_bits` - Number of parity bits (r)
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `HammingCode` instance or an error if parameters are invalid.
    pub fn with_params(
        parity_bits: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        if parity_bits < 2 {
            return Err(Error::InvalidInput(
                "Number of parity bits must be at least 2".into(),
            ));
        }

        let codeword_length = (1 << parity_bits) - 1;
        let message_length = codeword_length - parity_bits;

        Ok(Self {
            parity_bits,
            codeword_length,
            message_length,
            hardware_accelerator,
        })
    }

    /// Calculates the parity bits for a message.
    ///
    /// # Arguments
    ///
    /// * `message` - The message bits
    ///
    /// # Returns
    ///
    /// The parity bits.
    fn calculate_parity_bits(&self, message: &[u8]) -> Result<Vec<u8>> {
        let mut parity_bits = vec![0; self.parity_bits];

        // For each message bit, update the corresponding parity bits
        for (i, &bit) in message.iter().enumerate() {
            // Skip if bit is 0
            if bit == 0 {
                continue;
            }

            // Calculate the position in the codeword
            let pos = i + self.parity_bits + 1;

            // Update parity bits
            for j in 0..self.parity_bits {
                if (pos & (1 << j)) != 0 {
                    parity_bits[j] ^= 1;
                }
            }
        }

        Ok(parity_bits)
    }

    /// Calculates the syndrome for a received codeword.
    ///
    /// # Arguments
    ///
    /// * `codeword` - The received codeword
    ///
    /// # Returns
    ///
    /// The syndrome.
    fn calculate_syndrome(&self, codeword: &[u8]) -> Result<usize> {
        let mut syndrome = 0;

        // For each bit in the codeword
        for i in 0..codeword.len() {
            // Skip if bit is 0
            if codeword[i] == 0 {
                continue;
            }

            // Calculate the position (1-indexed)
            let pos = i + 1;

            // Update syndrome
            for j in 0..self.parity_bits {
                if (pos & (1 << j)) != 0 {
                    syndrome ^= 1 << j;
                }
            }
        }

        Ok(syndrome)
    }
}

impl ErrorCorrectionAlgorithm for HammingCode {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::HammingCode
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check if data size is valid
        if data.len() * 8 > self.message_length {
            return Err(Error::InvalidInput(
                format!(
                    "Input data too large: {} bytes (max: {} bits)",
                    data.len(),
                    self.message_length
                )
                .into(),
            ));
        }

        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_hamming()
        {
            return self
                .hardware_accelerator
                .hamming_encode(data, self.parity_bits)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Convert data to bits
        let mut message_bits = Vec::with_capacity(data.len() * 8);
        for &byte in data {
            for i in 0..8 {
                message_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Pad message if needed
        message_bits.resize(self.message_length, 0);

        // Calculate parity bits
        let parity_bits = self.calculate_parity_bits(&message_bits)?;

        // Construct codeword
        let mut codeword = vec![0; self.codeword_length];

        // Place parity bits at positions 1, 2, 4, 8, ...
        for i in 0..self.parity_bits {
            codeword[(1 << i) - 1] = parity_bits[i];
        }

        // Place message bits at other positions
        let mut msg_idx = 0;
        for i in 0..self.codeword_length {
            // Skip parity bit positions
            if !((i + 1) & (i + 1 - 1)).is_power_of_two() {
                codeword[i] = message_bits[msg_idx];
                msg_idx += 1;
            }
        }

        // Convert bits back to bytes
        let mut encoded = Vec::with_capacity((self.codeword_length + 7) / 8);
        for chunk in codeword.chunks(8) {
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

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Check if data size is valid
        if data.len() * 8 < self.codeword_length {
            return Err(Error::InvalidInput(
                format!(
                    "Input data too small: {} bytes (min: {} bits)",
                    data.len(),
                    self.codeword_length
                )
                .into(),
            ));
        }

        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_hamming()
        {
            return self
                .hardware_accelerator
                .hamming_decode(data, self.message_length)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Convert data to bits
        let mut codeword = Vec::with_capacity(data.len() * 8);
        for &byte in data {
            for i in 0..8 {
                codeword.push((byte >> (7 - i)) & 1);
            }
        }

        // Truncate to codeword length
        codeword.truncate(self.codeword_length);

        // Calculate syndrome
        let syndrome = self.calculate_syndrome(&codeword)?;

        // If syndrome is non-zero, correct the error
        if syndrome == 0 {
            // No errors
        } else if syndrome <= codeword.len() {
            // Single error, can be corrected
            codeword[syndrome - 1] ^= 1;
        } else {
            return Err(Error::TooManyErrors {
                detected: 2,
                correctable: 1,
            });
        }

        // Extract message bits
        let mut message_bits = Vec::with_capacity(self.message_length);
        for i in 0..self.codeword_length {
            // Skip parity bit positions
            if !((i + 1) & (i + 1 - 1)).is_power_of_two() {
                message_bits.push(codeword[i]);
            }
        }

        // Convert bits back to bytes
        let mut decoded = Vec::with_capacity((self.message_length + 7) / 8);
        for chunk in message_bits.chunks(8) {
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

    fn max_correctable_errors(&self) -> usize {
        1 // Hamming codes can correct 1 bit error
    }

    fn overhead_ratio(&self) -> f64 {
        self.codeword_length as f64 / self.message_length as f64
    }

    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the Hamming directory
        let hamming_path = path.join("hamming");
        std::fs::create_dir_all(&hamming_path)?;

        // Generate syndrome lookup table
        let mut syndrome_table = Vec::with_capacity(self.codeword_length);

        for i in 0..self.codeword_length {
            let mut codeword = vec![0; self.codeword_length];
            codeword[i] = 1; // Single error at position i

            let syndrome = self.calculate_syndrome(&codeword)?;
            syndrome_table.push((i, syndrome));
        }

        // Save syndrome table to file
        let syndrome_path = hamming_path.join("syndrome_table.bin");
        let syndrome_data = bincode::serialize(&syndrome_table).map_err(|_| {
            Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                "Failed to serialize syndrome table".to_string(),
            )))
        })?;

        std::fs::write(syndrome_path, syndrome_data)?;

        Ok(())
    }

    fn supports_hardware_acceleration(&self) -> bool {
        self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_hamming()
    }

    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator;
    }

    fn name(&self) -> &str {
        "Hamming Code"
    }
}

impl fmt::Debug for HammingCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HammingCode")
            .field("parity_bits", &self.parity_bits)
            .field("codeword_length", &self.codeword_length)
            .field("message_length", &self.message_length)
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .finish()
    }
}

/// Extended Hamming code implementation that can detect up to 2 errors.
pub struct ExtendedHammingCode {
    /// Base Hamming code
    base_hamming: HammingCode,
    /// The hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
}

impl ExtendedHammingCode {
    /// Creates a new Extended Hamming code instance.
    ///
    /// # Arguments
    ///
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `ExtendedHammingCode` instance or an error if creation failed.
    pub fn new(hardware_accelerator: Arc<dyn HardwareAccelerator>) -> Result<Self> {
        // Default to (8,4) Extended Hamming code
        Self::with_params(3, hardware_accelerator)
    }

    /// Creates a new Extended Hamming code instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `parity_bits` - Number of parity bits (r)
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `ExtendedHammingCode` instance or an error if parameters are invalid.
    pub fn with_params(
        parity_bits: usize,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        let base_hamming = HammingCode::with_params(parity_bits, hardware_accelerator.clone())?;

        Ok(Self {
            base_hamming,
            hardware_accelerator,
        })
    }

    /// Calculates the overall parity bit for a codeword.
    ///
    /// # Arguments
    ///
    /// * `codeword` - The codeword
    ///
    /// # Returns
    ///
    /// The overall parity bit.
    fn calculate_overall_parity(&self, codeword: &[u8]) -> u8 {
        let mut parity = 0;
        for &bit in codeword {
            parity ^= bit;
        }
        parity
    }
}

impl ErrorCorrectionAlgorithm for ExtendedHammingCode {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::ExtendedHammingCode
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Encode using base Hamming code
        let encoded = self.base_hamming.encode(data)?;

        // Convert to bits
        let mut codeword_bits = Vec::with_capacity(encoded.len() * 8);
        for &byte in &encoded {
            for i in 0..8 {
                codeword_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Truncate to codeword length
        codeword_bits.truncate(self.base_hamming.codeword_length);

        // Calculate overall parity bit
        let overall_parity = self.calculate_overall_parity(&codeword_bits);

        // Add overall parity bit
        codeword_bits.push(overall_parity);

        // Convert bits back to bytes
        let mut extended_encoded = Vec::with_capacity((codeword_bits.len() + 7) / 8);
        for chunk in codeword_bits.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if i < 8 {
                    byte |= bit << (7 - i);
                }
            }
            extended_encoded.push(byte);
        }

        Ok(extended_encoded)
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Convert data to bits
        let mut codeword_bits = Vec::with_capacity(data.len() * 8);
        for &byte in data {
            for i in 0..8 {
                codeword_bits.push((byte >> (7 - i)) & 1);
            }
        }

        // Ensure we have enough bits
        if codeword_bits.len() < self.base_hamming.codeword_length + 1 {
            return Err(Error::InvalidInput(
                format!(
                    "Input data too small: {} bits (min: {} bits)",
                    codeword_bits.len(),
                    self.base_hamming.codeword_length + 1
                )
                .into(),
            ));
        }

        // Extract overall parity bit
        let overall_parity_bit = codeword_bits[self.base_hamming.codeword_length];

        // Extract Hamming codeword
        let hamming_codeword = &codeword_bits[0..self.base_hamming.codeword_length];

        // Calculate overall parity
        let calculated_parity = self.calculate_overall_parity(hamming_codeword);

        // Calculate syndrome
        let syndrome = self.base_hamming.calculate_syndrome(hamming_codeword)?;

        // Error detection and correction
        let mut corrected_codeword = hamming_codeword.to_vec();

        if syndrome == 0 && calculated_parity == overall_parity_bit {
            // No errors
        } else if syndrome != 0 && calculated_parity != overall_parity_bit {
            // Single error, can be corrected
            if syndrome <= hamming_codeword.len() {
                corrected_codeword[syndrome - 1] ^= 1;
            } else {
                return Err(Error::TooManyErrors {
                    detected: 1,
                    correctable: 1,
                });
            }
        } else {
            // Double error, cannot be corrected
            return Err(Error::TooManyErrors {
                detected: 2,
                correctable: 1,
            });
        }

        // Convert corrected codeword to bytes for base Hamming decoding
        let mut corrected_bytes = Vec::with_capacity((corrected_codeword.len() + 7) / 8);
        for chunk in corrected_codeword.chunks(8) {
            let mut byte = 0;
            for (i, &bit) in chunk.iter().enumerate() {
                if i < 8 {
                    byte |= bit << (7 - i);
                }
            }
            corrected_bytes.push(byte);
        }

        // Decode using base Hamming code
        self.base_hamming.decode(&corrected_bytes)
    }

    fn max_correctable_errors(&self) -> usize {
        1 // Extended Hamming codes can correct 1 bit error and detect 2 bit errors
    }

    fn overhead_ratio(&self) -> f64 {
        (self.base_hamming.codeword_length as f64 + 1.0) / self.base_hamming.message_length as f64
    }

    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the ExtendedHamming directory
        let ext_hamming_path = path.join("extended_hamming");
        std::fs::create_dir_all(&ext_hamming_path)?;

        // Generate lookup tables for the base implementation
        self.base_hamming
            .generate_lookup_tables(&ext_hamming_path)?;

        Ok(())
    }

    fn supports_hardware_acceleration(&self) -> bool {
        self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_hamming()
    }

    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator.clone();
        self.base_hamming.set_hardware_accelerator(accelerator);
    }

    fn name(&self) -> &str {
        "Extended Hamming Code"
    }
}

impl fmt::Debug for ExtendedHammingCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtendedHammingCode")
            .field("base_hamming", &self.base_hamming)
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .finish()
    }
}
