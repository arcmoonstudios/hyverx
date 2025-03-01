//! Fountain code implementation for error correction.
//!
//! Fountain codes are a class of erasure codes that produce a potentially limitless
//! stream of encoding symbols from a fixed set of source symbols. They are rateless
//! codes, meaning they can produce as many encoded symbols as needed.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use super::{AlgorithmType, ErrorCorrectionAlgorithm};
use crate::error::{Error, Result};
use crate::hardware::HardwareAccelerator;

/// Implementation of Fountain codes (specifically LT codes) for error correction.
pub struct FountainCode {
    /// The number of source symbols
    source_symbols: usize,
    /// The symbol size in bytes
    symbol_size: usize,
    /// The degree distribution
    degree_distribution: DegreeDistribution,
    /// The random number generator seed
    seed: u64,
    /// The hardware accelerator for optimized operations
    hardware_accelerator: Arc<dyn HardwareAccelerator>,
}

/// Degree distribution for fountain codes
#[derive(Debug)]
#[allow(dead_code)]
pub enum DegreeDistribution {
    /// Robust Soliton distribution
    RobustSoliton {
        /// The c parameter for the robust soliton distribution
        c: f64,
        /// The delta parameter for the robust soliton distribution
        delta: f64,
    },
    /// Ideal Soliton distribution
    IdealSoliton,
    /// Custom distribution
    Custom(Vec<f64>),
}

/// Encoded symbol with metadata
#[derive(Debug)]
struct EncodedSymbol {
    /// The encoded data
    data: Vec<u8>,
    /// The seed used to generate the neighbors
    #[allow(dead_code)]
    seed: u64,
    /// The degree (number of source symbols combined)
    #[allow(dead_code)]
    degree: usize,
    /// The indices of the source symbols combined
    neighbors: Vec<usize>,
}

impl FountainCode {
    /// Creates a new Fountain code instance with default parameters.
    ///
    /// # Arguments
    ///
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `FountainCode` instance or an error if creation failed.
    pub fn new(hardware_accelerator: Arc<dyn HardwareAccelerator>) -> Result<Self> {
        // Default parameters for a fountain code
        Self::with_params(
            1000,
            1024,
            DegreeDistribution::RobustSoliton {
                c: 0.03,
                delta: 0.05,
            },
            0,
            hardware_accelerator,
        )
    }

    /// Creates a new Fountain code instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `source_symbols` - The number of source symbols
    /// * `symbol_size` - The symbol size in bytes
    /// * `degree_distribution` - The degree distribution
    /// * `seed` - The random number generator seed
    /// * `hardware_accelerator` - Hardware accelerator for optimized operations
    ///
    /// # Returns
    ///
    /// A new `FountainCode` instance or an error if parameters are invalid.
    pub fn with_params(
        source_symbols: usize,
        symbol_size: usize,
        degree_distribution: DegreeDistribution,
        seed: u64,
        hardware_accelerator: Arc<dyn HardwareAccelerator>,
    ) -> Result<Self> {
        if source_symbols == 0 {
            return Err(Error::InvalidInput(
                "Number of source symbols must be positive".into(),
            ));
        }

        if symbol_size == 0 {
            return Err(Error::InvalidInput("Symbol size must be positive".into()));
        }

        Ok(Self {
            source_symbols,
            symbol_size,
            degree_distribution,
            seed,
            hardware_accelerator,
        })
    }

    /// Samples a degree from the degree distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - The random number generator
    ///
    /// # Returns
    ///
    /// A sampled degree.
    fn sample_degree(&self, rng: &mut StdRng) -> usize {
        let k = self.source_symbols;

        match &self.degree_distribution {
            DegreeDistribution::RobustSoliton { c, delta: _delta } => {
                // Calculate the robust soliton distribution
                let s = c * (k as f64).ln() * (k as f64).sqrt();
                let s = s as usize;

                // Generate the ideal soliton distribution
                let mut rho = vec![0.0; k + 1];
                rho[1] = 1.0 / k as f64;
                for i in 2..=k {
                    rho[i] = 1.0 / (i * (i - 1)) as f64;
                }

                // Generate the tau component
                let mut tau = vec![0.0; k + 1];
                for i in 1..=s {
                    tau[i] = s as f64 / (i * k) as f64;
                }
                tau[s] = s as f64 * (k as f64).ln() / k as f64;

                // Combine to get the robust soliton distribution
                let mut mu = vec![0.0; k + 1];
                let mut sum = 0.0;
                for i in 1..=k {
                    mu[i] = rho[i] + tau[i];
                    sum += mu[i];
                }

                // Normalize
                for i in 1..=k {
                    mu[i] /= sum;
                }

                // Sample from the distribution
                let u: f64 = rng.random();
                let mut cumsum = 0.0;
                for i in 1..=k {
                    cumsum += mu[i];
                    if u <= cumsum {
                        return i;
                    }
                }

                // Fallback
                1
            }
            DegreeDistribution::IdealSoliton => {
                // Generate the ideal soliton distribution
                let mut rho = vec![0.0; k + 1];
                rho[1] = 1.0 / k as f64;
                for i in 2..=k {
                    rho[i] = 1.0 / (i * (i - 1)) as f64;
                }

                // Normalize
                let sum: f64 = rho.iter().sum();
                for i in 1..=k {
                    rho[i] /= sum;
                }

                // Sample from the distribution
                let u: f64 = rng.random();
                let mut cumsum = 0.0;
                for i in 1..=k {
                    cumsum += rho[i];
                    if u <= cumsum {
                        return i;
                    }
                }

                // Fallback
                1
            }
            DegreeDistribution::Custom(dist) => {
                // Sample from the custom distribution
                let u: f64 = rng.random();
                let mut cumsum = 0.0;
                for (i, &p) in dist.iter().enumerate() {
                    cumsum += p;
                    if u <= cumsum {
                        return i + 1;
                    }
                }

                // Fallback
                1
            }
        }
    }

    /// Generates the neighbors for an encoded symbol.
    ///
    /// # Arguments
    ///
    /// * `degree` - The degree of the encoded symbol
    /// * `seed` - The seed for the random number generator
    ///
    /// # Returns
    ///
    /// The indices of the source symbols to combine.
    fn generate_neighbors(&self, degree: usize, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut neighbors = HashSet::new();

        while neighbors.len() < degree {
            let neighbor = rng.random_range(0..self.source_symbols);
            neighbors.insert(neighbor);
        }

        neighbors.into_iter().collect()
    }

    /// Encodes a message using the fountain encoder.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to encode
    /// * `num_symbols` - The number of encoded symbols to generate
    ///
    /// # Returns
    ///
    /// The encoded symbols.
    fn encode_message(&self, message: &[u8], num_symbols: usize) -> Result<Vec<Vec<u8>>> {
        // Check if message size is valid
        if message.len() > self.source_symbols * self.symbol_size {
            return Err(Error::InvalidInput(
                format!(
                    "Message too large: {} bytes (max: {} bytes)",
                    message.len(),
                    self.source_symbols * self.symbol_size
                )
                .into(),
            ));
        }

        // Pad message if needed
        let mut padded_message = message.to_vec();
        padded_message.resize(self.source_symbols * self.symbol_size, 0);

        // Split message into source symbols
        let mut source_symbols = Vec::with_capacity(self.source_symbols);
        for i in 0..self.source_symbols {
            let start = i * self.symbol_size;
            let end = start + self.symbol_size;
            source_symbols.push(padded_message[start..end].to_vec());
        }

        // Generate encoded symbols
        let mut encoded_symbols = Vec::with_capacity(num_symbols);
        let mut rng = StdRng::seed_from_u64(self.seed);

        for i in 0..num_symbols {
            // Generate a new seed for this symbol
            let symbol_seed = rng.random();

            // Sample a degree
            let degree = self.sample_degree(&mut rng);

            // Generate neighbors
            let neighbors = self.generate_neighbors(degree, symbol_seed);

            // Combine source symbols
            let mut data = vec![0; self.symbol_size];
            for &neighbor in &neighbors {
                for j in 0..self.symbol_size {
                    data[j] ^= source_symbols[neighbor][j];
                }
            }

            // Add metadata to the encoded symbol
            let mut encoded_symbol = Vec::with_capacity(self.symbol_size + 16);

            // Add symbol seed (8 bytes)
            encoded_symbol.extend_from_slice(&symbol_seed.to_le_bytes());

            // Add degree (4 bytes)
            encoded_symbol.extend_from_slice(&(degree as u32).to_le_bytes());

            // Add symbol index (4 bytes)
            encoded_symbol.extend_from_slice(&(i as u32).to_le_bytes());

            // Add data
            encoded_symbol.extend_from_slice(&data);

            encoded_symbols.push(encoded_symbol);
        }

        Ok(encoded_symbols)
    }

    /// Decodes a set of encoded symbols using the belief propagation algorithm.
    ///
    /// # Arguments
    ///
    /// * `encoded_symbols` - The encoded symbols to decode
    ///
    /// # Returns
    ///
    /// The decoded message.
    fn decode_symbols(&self, encoded_symbols: &[Vec<u8>]) -> Result<Vec<u8>> {
        // Extract metadata and data from encoded symbols
        let mut symbols = Vec::with_capacity(encoded_symbols.len());

        for encoded_symbol in encoded_symbols {
            if encoded_symbol.len() < 16 + self.symbol_size {
                return Err(Error::InvalidInput(
                    format!(
                        "Encoded symbol too small: {} bytes (min: {} bytes)",
                        encoded_symbol.len(),
                        16 + self.symbol_size
                    )
                    .into(),
                ));
            }

            // Extract symbol seed (8 bytes)
            let mut seed_bytes = [0; 8];
            seed_bytes.copy_from_slice(&encoded_symbol[0..8]);
            let seed = u64::from_le_bytes(seed_bytes);

            // Extract degree (4 bytes)
            let mut degree_bytes = [0; 4];
            degree_bytes.copy_from_slice(&encoded_symbol[8..12]);
            let degree = u32::from_le_bytes(degree_bytes) as usize;

            // Extract symbol index (4 bytes)
            let mut index_bytes = [0; 4];
            index_bytes.copy_from_slice(&encoded_symbol[12..16]);
            let _index = u32::from_le_bytes(index_bytes) as usize;

            // Extract data
            let data = encoded_symbol[16..].to_vec();

            // Generate neighbors
            let neighbors = self.generate_neighbors(degree, seed);

            symbols.push(EncodedSymbol {
                data,
                seed,
                degree,
                neighbors,
            });
        }

        // Initialize decoded source symbols
        let mut decoded = vec![None; self.source_symbols];
        let mut ripple = HashSet::new();

        // Belief propagation algorithm
        loop {
            // Find symbols of degree 1
            for symbol in &symbols {
                if symbol.neighbors.len() == 1 {
                    ripple.insert(symbol.neighbors[0]);
                }
            }

            if ripple.is_empty() {
                break;
            }

            // Process symbols in the ripple
            let neighbor = *ripple.iter().next().unwrap();
            ripple.remove(&neighbor);

            // Find a symbol that has this neighbor
            let mut symbol_idx = None;
            let mut symbol_data = Vec::new();

            for (i, symbol) in symbols.iter().enumerate() {
                if symbol.neighbors.len() == 1 && symbol.neighbors[0] == neighbor {
                    symbol_idx = Some(i);
                    symbol_data = symbol.data.clone();
                    break;
                }
            }

            if let Some(_idx) = symbol_idx {
                // Decode the source symbol
                decoded[neighbor] = Some(symbol_data.clone());

                // Update other symbols
                for other_symbol in &mut symbols {
                    if other_symbol.neighbors.contains(&neighbor) {
                        // Remove the neighbor
                        other_symbol.neighbors.retain(|&n| n != neighbor);

                        // XOR the data
                        for j in 0..self.symbol_size {
                            other_symbol.data[j] ^= symbol_data[j];
                        }
                    }
                }
            }
        }

        // Check if all source symbols are decoded
        if decoded.iter().any(|s| s.is_none()) {
            return Err(Error::Decoding(
                "Not enough encoded symbols to decode the message".to_string(),
            ));
        }

        // Combine decoded source symbols
        let mut message = Vec::with_capacity(self.source_symbols * self.symbol_size);
        for symbol in decoded {
            message.extend_from_slice(&symbol.unwrap());
        }

        Ok(message)
    }
}

impl ErrorCorrectionAlgorithm for FountainCode {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::Fountain
    }

    fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_fountain()
        {
            return self
                .hardware_accelerator
                .fountain_encode(data, self.source_symbols, self.symbol_size)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Software implementation
        // Generate 20% more symbols than source symbols for redundancy
        let num_symbols = (self.source_symbols * 12) / 10;
        let encoded_symbols = self.encode_message(data, num_symbols)?;

        // Flatten encoded symbols into a single vector
        let mut encoded = Vec::with_capacity(encoded_symbols.len() * (self.symbol_size + 16));
        for symbol in encoded_symbols {
            encoded.extend_from_slice(&symbol);
        }

        Ok(encoded)
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Use hardware acceleration if available
        if self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_fountain()
        {
            return self
                .hardware_accelerator
                .fountain_decode(data, self.source_symbols, self.symbol_size)
                .map_err(|e| Error::HardwareAcceleration(e.to_string()));
        }

        // Software implementation
        // Split data into encoded symbols
        let symbol_size_with_metadata = self.symbol_size + 16;
        let num_symbols = data.len() / symbol_size_with_metadata;

        if data.len() % symbol_size_with_metadata != 0 {
            return Err(Error::InvalidInput(
                format!(
                    "Data size ({}) is not a multiple of symbol size with metadata ({})",
                    data.len(),
                    symbol_size_with_metadata
                )
                .into(),
            ));
        }

        let mut encoded_symbols = Vec::with_capacity(num_symbols);
        for i in 0..num_symbols {
            let start = i * symbol_size_with_metadata;
            let end = start + symbol_size_with_metadata;
            encoded_symbols.push(data[start..end].to_vec());
        }

        // Decode symbols
        let decoded = self.decode_symbols(&encoded_symbols)?;

        Ok(decoded)
    }

    fn max_correctable_errors(&self) -> usize {
        // Fountain codes can recover from any pattern of erasures
        // as long as enough encoded symbols are received
        self.source_symbols / 5 // Can lose up to 20% of symbols
    }

    fn overhead_ratio(&self) -> f64 {
        // The overhead is the ratio of encoded symbols to source symbols
        // For fountain codes, this is typically around 1.1-1.2
        1.2
    }

    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the Fountain directory
        let fountain_path = path.join("fountain");
        std::fs::create_dir_all(&fountain_path)?;

        // Save degree distribution
        let dist_path = fountain_path.join("degree_distribution.bin");
        let dist_data = match &self.degree_distribution {
            DegreeDistribution::RobustSoliton { c, delta } => {
                bincode::serialize(&("robust_soliton", *c, *delta)).map_err(|_| {
                    Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                        "Failed to serialize robust soliton distribution".to_string(),
                    )))
                })?
            }
            DegreeDistribution::IdealSoliton => bincode::serialize(&("ideal_soliton", 0.0, 0.0))
                .map_err(|_| {
                    Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                        "Failed to serialize ideal soliton distribution".to_string(),
                    )))
                })?,
            DegreeDistribution::Custom(dist) => {
                bincode::serialize(&("custom", dist)).map_err(|_| {
                    Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                        "Failed to serialize custom distribution".to_string(),
                    )))
                })?
            }
        };

        std::fs::write(dist_path, dist_data)?;

        // Save parameters
        let params_path = fountain_path.join("parameters.bin");
        let params_data = bincode::serialize(&(self.source_symbols, self.symbol_size, self.seed))
            .map_err(|_| {
            Error::BinarySerialization(bincode::Error::new(bincode::ErrorKind::Custom(
                "Failed to serialize fountain parameters".to_string(),
            )))
        })?;

        std::fs::write(params_path, params_data)?;

        Ok(())
    }

    fn supports_hardware_acceleration(&self) -> bool {
        self.hardware_accelerator.is_available() && self.hardware_accelerator.supports_fountain()
    }

    fn set_hardware_accelerator(&mut self, accelerator: Arc<dyn HardwareAccelerator>) {
        self.hardware_accelerator = accelerator;
    }

    fn name(&self) -> &str {
        "Fountain Code"
    }
}

impl fmt::Debug for FountainCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FountainCode")
            .field("source_symbols", &self.source_symbols)
            .field("symbol_size", &self.symbol_size)
            .field(
                "degree_distribution",
                &match self.degree_distribution {
                    DegreeDistribution::RobustSoliton { c, delta } => {
                        format!("RobustSoliton(c={}, delta={})", c, delta)
                    }
                    DegreeDistribution::IdealSoliton => "IdealSoliton".to_string(),
                    DegreeDistribution::Custom(_) => "Custom".to_string(),
                },
            )
            .field("max_correctable_errors", &self.max_correctable_errors())
            .field("overhead_ratio", &self.overhead_ratio())
            .finish()
    }
}
