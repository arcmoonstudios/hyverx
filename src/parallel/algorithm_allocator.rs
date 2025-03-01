//! Algorithm allocator for distributing error correction algorithms.
//!
//! This module provides utilities for allocating error correction algorithms
//! to different error patterns and regions for optimal parallel processing.

use std::sync::Arc;
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::algorithms::AlgorithmType;
use super::{Region, WorkItem, WorkScheduler};

/// An algorithm allocator for distributing error correction algorithms.
#[derive(Debug)]
pub struct AlgorithmAllocator {
    /// The work scheduler for executing tasks
    scheduler: Arc<WorkScheduler>,
    /// The available algorithms
    algorithms: Vec<AlgorithmType>,
    /// The allocation strategy
    strategy: AllocationStrategy,
    /// The allocation map
    allocation_map: HashMap<usize, AlgorithmType>,
}

/// A strategy for allocating algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Allocate algorithms based on error pattern
    ErrorPattern,
    /// Allocate algorithms based on region size
    RegionSize,
    /// Allocate algorithms based on error density
    ErrorDensity,
    /// Allocate algorithms based on a fixed pattern
    Fixed,
    /// Allocate algorithms randomly
    Random,
}

/// An error pattern descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorPattern {
    /// The region of the error pattern
    pub region: Region,
    /// The error density (errors per unit)
    pub error_density: f64,
    /// The error distribution type
    pub distribution: ErrorDistribution,
    /// The error burst length (if applicable)
    pub burst_length: Option<usize>,
}

/// An error distribution type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorDistribution {
    /// Random errors
    Random,
    /// Burst errors
    Burst,
    /// Clustered errors
    Clustered,
    /// Periodic errors
    Periodic,
}

impl AlgorithmAllocator {
    /// Creates a new algorithm allocator with the given scheduler.
    ///
    /// # Arguments
    ///
    /// * `scheduler` - The work scheduler for executing tasks
    ///
    /// # Returns
    ///
    /// A new `AlgorithmAllocator` instance.
    pub fn new(scheduler: Arc<WorkScheduler>) -> Self {
        Self {
            scheduler,
            algorithms: vec![
                AlgorithmType::ReedSolomon,
                AlgorithmType::Ldpc,
                AlgorithmType::Turbo,
                AlgorithmType::PolarCode,
            ],
            strategy: AllocationStrategy::ErrorPattern,
            allocation_map: HashMap::new(),
        }
    }
    
    /// Sets the available algorithms.
    ///
    /// # Arguments
    ///
    /// * `algorithms` - The available algorithms
    ///
    /// # Returns
    ///
    /// A reference to the updated `AlgorithmAllocator` instance.
    pub fn with_algorithms(&mut self, algorithms: Vec<AlgorithmType>) -> &mut Self {
        self.algorithms = algorithms;
        self
    }
    
    /// Sets the allocation strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The allocation strategy to use
    ///
    /// # Returns
    ///
    /// A reference to the updated `AlgorithmAllocator` instance.
    pub fn with_strategy(&mut self, strategy: AllocationStrategy) -> &mut Self {
        self.strategy = strategy;
        self
    }
    
    /// Allocates algorithms to error patterns.
    ///
    /// # Arguments
    ///
    /// * `error_patterns` - The error patterns to allocate algorithms to
    ///
    /// # Returns
    ///
    /// A vector of work items with allocated algorithms.
    pub fn allocate(&mut self, error_patterns: &[ErrorPattern]) -> Result<Vec<WorkItem>> {
        if self.algorithms.is_empty() {
            return Err(Error::InvalidInput("No algorithms available".into()));
        }
        
        let mut work_items = Vec::with_capacity(error_patterns.len());
        
        for (i, pattern) in error_patterns.iter().enumerate() {
            let algorithm = self.select_algorithm(pattern)?;
            
            work_items.push(WorkItem {
                region: pattern.region.clone(),
                algorithm,
                priority: i,
            });
            
            // Store the allocation for future reference
            self.allocation_map.insert(i, algorithm);
        }
        
        Ok(work_items)
    }
    
    /// Selects an algorithm for an error pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The error pattern to select an algorithm for
    ///
    /// # Returns
    ///
    /// The selected algorithm.
    fn select_algorithm(&self, pattern: &ErrorPattern) -> Result<AlgorithmType> {
        match self.strategy {
            AllocationStrategy::ErrorPattern => {
                // Select algorithm based on error pattern
                match pattern.distribution {
                    ErrorDistribution::Random => {
                        // For random errors, use LDPC or Reed-Solomon
                        if pattern.error_density < 0.1 {
                            // For low error density, use LDPC
                            if self.algorithms.contains(&AlgorithmType::Ldpc) {
                                Ok(AlgorithmType::Ldpc)
                            } else {
                                // Fall back to Reed-Solomon
                                Ok(AlgorithmType::ReedSolomon)
                            }
                        } else {
                            // For high error density, use Reed-Solomon
                            Ok(AlgorithmType::ReedSolomon)
                        }
                    },
                    ErrorDistribution::Burst => {
                        // For burst errors, use Reed-Solomon
                        Ok(AlgorithmType::ReedSolomon)
                    },
                    ErrorDistribution::Clustered => {
                        // For clustered errors, use Turbo codes
                        if self.algorithms.contains(&AlgorithmType::Turbo) {
                            Ok(AlgorithmType::Turbo)
                        } else {
                            // Fall back to Reed-Solomon
                            Ok(AlgorithmType::ReedSolomon)
                        }
                    },
                    ErrorDistribution::Periodic => {
                        // For periodic errors, use Polar codes
                        if self.algorithms.contains(&AlgorithmType::PolarCode) {
                            Ok(AlgorithmType::PolarCode)
                        } else {
                            // Fall back to Reed-Solomon
                            Ok(AlgorithmType::ReedSolomon)
                        }
                    },
                }
            },
            AllocationStrategy::RegionSize => {
                // Select algorithm based on region size
                let region_size = pattern.region.size();
                
                if region_size < 1000 {
                    // For small regions, use Reed-Solomon
                    Ok(AlgorithmType::ReedSolomon)
                } else if region_size < 10000 {
                    // For medium regions, use LDPC
                    if self.algorithms.contains(&AlgorithmType::Ldpc) {
                        Ok(AlgorithmType::Ldpc)
                    } else {
                        // Fall back to Reed-Solomon
                        Ok(AlgorithmType::ReedSolomon)
                    }
                } else {
                    // For large regions, use Turbo codes
                    if self.algorithms.contains(&AlgorithmType::Turbo) {
                        Ok(AlgorithmType::Turbo)
                    } else {
                        // Fall back to Reed-Solomon
                        Ok(AlgorithmType::ReedSolomon)
                    }
                }
            },
            AllocationStrategy::ErrorDensity => {
                // Select algorithm based on error density
                if pattern.error_density < 0.01 {
                    // For very low error density, use Polar codes
                    if self.algorithms.contains(&AlgorithmType::PolarCode) {
                        Ok(AlgorithmType::PolarCode)
                    } else {
                        // Fall back to LDPC
                        if self.algorithms.contains(&AlgorithmType::Ldpc) {
                            Ok(AlgorithmType::Ldpc)
                        } else {
                            // Fall back to Reed-Solomon
                            Ok(AlgorithmType::ReedSolomon)
                        }
                    }
                } else if pattern.error_density < 0.1 {
                    // For low error density, use LDPC
                    if self.algorithms.contains(&AlgorithmType::Ldpc) {
                        Ok(AlgorithmType::Ldpc)
                    } else {
                        // Fall back to Reed-Solomon
                        Ok(AlgorithmType::ReedSolomon)
                    }
                } else if pattern.error_density < 0.2 {
                    // For medium error density, use Turbo codes
                    if self.algorithms.contains(&AlgorithmType::Turbo) {
                        Ok(AlgorithmType::Turbo)
                    } else {
                        // Fall back to Reed-Solomon
                        Ok(AlgorithmType::ReedSolomon)
                    }
                } else {
                    // For high error density, use Reed-Solomon
                    Ok(AlgorithmType::ReedSolomon)
                }
            },
            AllocationStrategy::Fixed => {
                // Always use the first available algorithm
                Ok(self.algorithms[0])
            },
            AllocationStrategy::Random => {
                // Use a random algorithm
                use rand::Rng;
                let mut rng = rand::rng();
                let index = rng.random_range(0..self.algorithms.len());
                Ok(self.algorithms[index])
            },
        }
    }
    
    /// Schedules the allocated work items.
    ///
    /// # Arguments
    ///
    /// * `work_items` - The work items to schedule
    ///
    /// # Returns
    ///
    /// `Ok(())` if the scheduling was successful, or an error if scheduling failed.
    pub fn schedule(&self, work_items: Vec<WorkItem>) -> Result<()> {
        self.scheduler.schedule_batch(work_items)?;
        Ok(())
    }
    
    /// Allocates algorithms to error patterns and schedules the work items.
    ///
    /// # Arguments
    ///
    /// * `error_patterns` - The error patterns to allocate algorithms to
    ///
    /// # Returns
    ///
    /// `Ok(())` if the allocation and scheduling were successful, or an error if they failed.
    pub fn allocate_and_schedule(&mut self, error_patterns: &[ErrorPattern]) -> Result<()> {
        let work_items = self.allocate(error_patterns)?;
        self.schedule(work_items)?;
        Ok(())
    }
    
    /// Waits for all scheduled work items to complete.
    ///
    /// # Returns
    ///
    /// `Ok(())` if waiting was successful, or an error if waiting failed.
    pub fn wait(&self) -> Result<()> {
        self.scheduler.wait()?;
        Ok(())
    }
    
    /// Returns the allocation map.
    pub fn allocation_map(&self) -> &HashMap<usize, AlgorithmType> {
        &self.allocation_map
    }
    
    /// Selects an encoding algorithm based on data characteristics.
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to encode
    /// * `error_rate` - Expected error rate
    /// * `burst_errors` - Whether burst errors are expected
    ///
    /// # Returns
    ///
    /// The selected algorithm type.
    pub fn select_encoding_algorithm(
        &self, 
        _data_size: usize, 
        error_rate: f64, 
        burst_errors: bool
    ) -> Result<AlgorithmType> {
        // Simple algorithm selection logic based on data characteristics
        if burst_errors {
            if self.algorithms.contains(&AlgorithmType::ReedSolomon) {
                return Ok(AlgorithmType::ReedSolomon);
            }
        }
        
        if error_rate < 0.01 {
            // For very low error rates, use LDPC or Polar codes
            if self.algorithms.contains(&AlgorithmType::Ldpc) {
                return Ok(AlgorithmType::Ldpc);
            } else if self.algorithms.contains(&AlgorithmType::PolarCode) {
                return Ok(AlgorithmType::PolarCode);
            }
        } else if error_rate < 0.05 {
            // For moderate error rates, use Reed-Solomon
            if self.algorithms.contains(&AlgorithmType::ReedSolomon) {
                return Ok(AlgorithmType::ReedSolomon);
            }
        } else {
            // For high error rates, use more robust algorithms
            if self.algorithms.contains(&AlgorithmType::TensorReedSolomon) {
                return Ok(AlgorithmType::TensorReedSolomon);
            } else if self.algorithms.contains(&AlgorithmType::AdaptiveReedSolomon) {
                return Ok(AlgorithmType::AdaptiveReedSolomon);
            }
        }
        
        // Default to Reed-Solomon if available
        if self.algorithms.contains(&AlgorithmType::ReedSolomon) {
            return Ok(AlgorithmType::ReedSolomon);
        }
        
        // If no algorithms are available, return an error
        Err(Error::InvalidInput("No suitable encoding algorithm available".into()))
    }
    
    /// Selects a decoding algorithm based on data characteristics.
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to decode
    /// * `error_rate` - Expected error rate
    /// * `burst_errors` - Whether burst errors are expected
    ///
    /// # Returns
    ///
    /// The selected algorithm type.
    pub fn select_decoding_algorithm(
        &self, 
        _data_size: usize, 
        error_rate: f64, 
        burst_errors: bool
    ) -> Result<AlgorithmType> {
        // For decoding, we should use the same algorithm that was used for encoding
        // But we can optimize based on the expected error patterns
        
        if burst_errors {
            if self.algorithms.contains(&AlgorithmType::ReedSolomon) {
                return Ok(AlgorithmType::ReedSolomon);
            }
        }
        
        if error_rate < 0.01 {
            // For very low error rates, use LDPC or Polar codes
            if self.algorithms.contains(&AlgorithmType::Ldpc) {
                return Ok(AlgorithmType::Ldpc);
            } else if self.algorithms.contains(&AlgorithmType::PolarCode) {
                return Ok(AlgorithmType::PolarCode);
            }
        } else if error_rate < 0.05 {
            // For moderate error rates, use Reed-Solomon
            if self.algorithms.contains(&AlgorithmType::ReedSolomon) {
                return Ok(AlgorithmType::ReedSolomon);
            }
        } else {
            // For high error rates, use more robust algorithms
            if self.algorithms.contains(&AlgorithmType::TensorReedSolomon) {
                return Ok(AlgorithmType::TensorReedSolomon);
            } else if self.algorithms.contains(&AlgorithmType::AdaptiveReedSolomon) {
                return Ok(AlgorithmType::AdaptiveReedSolomon);
            }
        }
        
        // Default to Reed-Solomon if available
        if self.algorithms.contains(&AlgorithmType::ReedSolomon) {
            return Ok(AlgorithmType::ReedSolomon);
        }
        
        // If no algorithms are available, return an error
        Err(Error::InvalidInput("No suitable decoding algorithm available".into()))
    }
    
    /// Creates a new error pattern.
    ///
    /// # Arguments
    ///
    /// * `region` - The region of the error pattern
    /// * `error_density` - The error density (errors per unit)
    /// * `distribution` - The error distribution type
    /// * `burst_length` - The error burst length (if applicable)
    ///
    /// # Returns
    ///
    /// A new `ErrorPattern` instance.
    pub fn create_error_pattern(
        region: Region,
        error_density: f64,
        distribution: ErrorDistribution,
        burst_length: Option<usize>,
    ) -> ErrorPattern {
        ErrorPattern {
            region,
            error_density,
            distribution,
            burst_length,
        }
    }
    
    /// Creates error patterns for a set of regions.
    ///
    /// # Arguments
    ///
    /// * `regions` - The regions to create error patterns for
    /// * `error_density` - The error density (errors per unit)
    /// * `distribution` - The error distribution type
    ///
    /// # Returns
    ///
    /// A vector of `ErrorPattern` instances.
    pub fn create_error_patterns(
        regions: &[Region],
        error_density: f64,
        distribution: ErrorDistribution,
    ) -> Vec<ErrorPattern> {
        regions.iter()
            .map(|region| {
                ErrorPattern {
                    region: region.clone(),
                    error_density,
                    distribution,
                    burst_length: None,
                }
            })
            .collect()
    }
} 