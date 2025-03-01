//! Dimension mapper implementation for multi-dimensional data processing.
//!
//! This module provides utilities for mapping multi-dimensional data to threads
//! for parallel processing. It includes features like dimension splitting, region
//! mapping, and efficient data access patterns.

use std::sync::Arc;
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::parallel::work_scheduler::WorkItemHandler;
use super::{Dimension, Region, WorkItem, WorkScheduler, ParallelConfig};

/// A dimension mapper for multi-dimensional data processing.
#[derive(Debug)]
pub struct DimensionMapper {
    /// The work scheduler for executing tasks
    scheduler: Arc<WorkScheduler>,
    /// The configuration for the mapper
    #[allow(dead_code)]
    config: ParallelConfig,
    /// The dimensions of the data
    dimensions: Vec<Dimension>,
    /// The regions of the data
    regions: Vec<Region>,
    /// The dimension splitting strategy
    splitting_strategy: SplittingStrategy,
}

/// A strategy for splitting dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplittingStrategy {
    /// Split along the largest dimension
    Largest,
    /// Split along the smallest dimension
    Smallest,
    /// Split along all dimensions evenly
    Even,
    /// Split along the first dimension
    First,
    /// Split along the last dimension
    Last,
    /// Split along a specific dimension
    Specific(usize),
}

impl DimensionMapper {
    /// Creates a new dimension mapper with the given scheduler and configuration.
    ///
    /// # Arguments
    ///
    /// * `scheduler` - The work scheduler for executing tasks
    /// * `config` - Configuration for the mapper
    ///
    /// # Returns
    ///
    /// A new `DimensionMapper` instance.
    pub fn new(scheduler: Arc<WorkScheduler>, config: ParallelConfig) -> Self {
        Self {
            scheduler,
            config,
            dimensions: Vec::new(),
            regions: Vec::new(),
            splitting_strategy: SplittingStrategy::Largest,
        }
    }
    
    /// Creates a new dimension mapper with the given scheduler and default configuration.
    ///
    /// # Arguments
    ///
    /// * `scheduler` - The work scheduler for executing tasks
    ///
    /// # Returns
    ///
    /// A new `DimensionMapper` instance.
    pub fn with_scheduler(scheduler: Arc<WorkScheduler>) -> Self {
        Self::new(scheduler, ParallelConfig::default())
    }
    
    /// Sets the dimensions of the data.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensions of the data
    ///
    /// # Returns
    ///
    /// A reference to the updated `DimensionMapper` instance.
    pub fn with_dimensions(&mut self, dimensions: Vec<Dimension>) -> &mut Self {
        self.dimensions = dimensions;
        self.regions = vec![Region::new(self.dimensions.clone())];
        self
    }
    
    /// Sets the splitting strategy for the mapper.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The splitting strategy to use
    ///
    /// # Returns
    ///
    /// A reference to the updated `DimensionMapper` instance.
    pub fn with_splitting_strategy(&mut self, strategy: SplittingStrategy) -> &mut Self {
        self.splitting_strategy = strategy;
        self
    }
    
    /// Splits the data into regions for parallel processing.
    ///
    /// # Arguments
    ///
    /// * `thread_count` - The number of threads to split for
    ///
    /// # Returns
    ///
    /// `Ok(())` if the data was split successfully, or an error if splitting failed.
    pub fn split_for_threads(&mut self, thread_count: usize) -> Result<&mut Self> {
        if self.dimensions.is_empty() {
            return Err(Error::InvalidInput("No dimensions defined".into()));
        }
        
        // Determine which dimension to split along
        let split_dim_index = match self.splitting_strategy {
            SplittingStrategy::Largest => {
                // Find the largest dimension
                let mut largest_index = 0;
                let mut largest_size = 0;
                
                for (i, dim) in self.dimensions.iter().enumerate() {
                    if dim.size > largest_size {
                        largest_index = i;
                        largest_size = dim.size;
                    }
                }
                
                largest_index
            },
            SplittingStrategy::Smallest => {
                // Find the smallest dimension
                let mut smallest_index = 0;
                let mut smallest_size = usize::MAX;
                
                for (i, dim) in self.dimensions.iter().enumerate() {
                    if dim.size < smallest_size {
                        smallest_index = i;
                        smallest_size = dim.size;
                    }
                }
                
                smallest_index
            },
            SplittingStrategy::Even => {
                // Split evenly across all dimensions
                // For now, just use the largest dimension
                // TODO: Implement even splitting
                let mut largest_index = 0;
                let mut largest_size = 0;
                
                for (i, dim) in self.dimensions.iter().enumerate() {
                    if dim.size > largest_size {
                        largest_index = i;
                        largest_size = dim.size;
                    }
                }
                
                largest_index
            },
            SplittingStrategy::First => 0,
            SplittingStrategy::Last => self.dimensions.len() - 1,
            SplittingStrategy::Specific(index) => {
                if index >= self.dimensions.len() {
                    return Err(Error::InvalidInput(
                        format!("Dimension index {} out of bounds (max: {})", index, self.dimensions.len() - 1).into(),
                    ));
                }
                
                index
            },
        };
        
        // Split the region
        let base_region = Region::new(self.dimensions.clone());
        self.regions = base_region.split(split_dim_index, thread_count);
        
        Ok(self)
    }
    
    /// Maps a function over the regions in parallel.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to map over the regions
    ///
    /// # Returns
    ///
    /// `Ok(())` if the mapping was successful, or an error if mapping failed.
    pub fn map<F>(&self, f: F) -> Result<()>
    where
        F: Fn(&Region) -> Result<()> + Send + Sync + 'static,
    {
        if self.regions.is_empty() {
            return Err(Error::InvalidInput("No regions defined".into()));
        }
        
        // Create a work item for each region
        let work_items: Vec<_> = self.regions.iter()
            .enumerate()
            .map(|(i, region)| {
                WorkItem {
                    region: region.clone(),
                    algorithm: crate::algorithms::AlgorithmType::ReedSolomon, // Default algorithm
                    priority: i,
                }
            })
            .collect();
        
        // Create a handler for the work items
        struct MapHandler<F: 'static> {
            f: Arc<F>,
        }
        
        impl<F> WorkItemHandler for MapHandler<F>
        where
            F: Fn(&Region) -> Result<()> + Send + Sync + 'static,
        {
            fn process(&self, item: &WorkItem) -> Result<()> {
                (self.f)(&item.region)
            }
            
            fn algorithm_type(&self) -> crate::algorithms::AlgorithmType {
                crate::algorithms::AlgorithmType::ReedSolomon // Default algorithm
            }

            fn clone_box(&self) -> Box<dyn WorkItemHandler + Send + Sync> {
                Box::new(MapHandler {
                    f: Arc::clone(&self.f),
                })
            }
        }
        
        // Register the handler
        self.scheduler.register_handler(MapHandler {
            f: Arc::new(f),
        })?;
        
        // Schedule the work items
        self.scheduler.schedule_batch(work_items)?;
        
        // Wait for all work items to complete
        self.scheduler.wait()?;
        
        Ok(())
    }
    
    /// Maps a function over the regions in parallel and collects the results.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to map over the regions
    ///
    /// # Returns
    ///
    /// A vector of results, one for each region.
    pub fn map_collect<F, T>(&self, f: F) -> Result<Vec<T>>
    where
        F: Fn(&Region) -> Result<T> + Send + Sync + 'static,
        T: Send + Sync + 'static + Clone,
    {
        if self.regions.is_empty() {
            return Err(Error::InvalidInput("No regions defined".into()));
        }
        
        // Create a map to store the results
        let results: Arc<parking_lot::Mutex<HashMap<usize, T>>> = Arc::new(parking_lot::Mutex::new(HashMap::new()));
        
        // Create a work item for each region
        let work_items: Vec<_> = self.regions.iter()
            .enumerate()
            .map(|(i, region)| {
                WorkItem {
                    region: region.clone(),
                    algorithm: crate::algorithms::AlgorithmType::ReedSolomon, // Default algorithm
                    priority: i,
                }
            })
            .collect();
        
        // Create a handler for the work items
        struct MapCollectHandler<F: 'static, T: 'static> {
            f: Arc<F>,
            results: Arc<parking_lot::Mutex<HashMap<usize, T>>>,
        }
        
        impl<F, T> WorkItemHandler for MapCollectHandler<F, T>
        where
            F: Fn(&Region) -> Result<T> + Send + Sync + 'static,
            T: Send + Sync + 'static + Clone,
        {
            fn process(&self, item: &WorkItem) -> Result<()> {
                let result = (self.f)(&item.region)?;
                self.results.lock().insert(item.priority, result);
                Ok(())
            }
            
            fn algorithm_type(&self) -> crate::algorithms::AlgorithmType {
                crate::algorithms::AlgorithmType::ReedSolomon // Default algorithm
            }

            fn clone_box(&self) -> Box<dyn WorkItemHandler + Send + Sync> {
                Box::new(MapCollectHandler {
                    f: Arc::clone(&self.f),
                    results: Arc::clone(&self.results),
                })
            }
        }
        
        // Register the handler
        self.scheduler.register_handler(MapCollectHandler {
            f: Arc::new(f),
            results: Arc::clone(&results),
        })?;
        
        // Schedule the work items
        self.scheduler.schedule_batch(work_items)?;
        
        // Wait for all work items to complete
        self.scheduler.wait()?;
        
        // Collect the results in order
        let results_map = results.lock();
        let mut results_vec = Vec::with_capacity(self.regions.len());
        
        for i in 0..self.regions.len() {
            if let Some(result) = results_map.get(&i).cloned() {
                results_vec.push(result);
            } else {
                return Err(Error::Internal(
                    format!("Missing result for region {}", i).into(),
                ));
            }
        }
        
        Ok(results_vec)
    }
    
    /// Returns the regions of the data.
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
    
    /// Returns the dimensions of the data.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }
    
    /// Returns the total size of the data.
    pub fn total_size(&self) -> usize {
        self.dimensions.iter()
            .map(|d| d.size)
            .product()
    }
    
    /// Creates a new dimension with the given size and stride.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the dimension
    /// * `size` - The size of the dimension
    /// * `stride` - The stride of the dimension
    ///
    /// # Returns
    ///
    /// A new `Dimension` instance.
    pub fn create_dimension(index: usize, size: usize, stride: usize) -> Dimension {
        Dimension {
            index,
            size,
            stride,
        }
    }
    
    /// Creates dimensions for a multi-dimensional array.
    ///
    /// # Arguments
    ///
    /// * `sizes` - The sizes of each dimension
    ///
    /// # Returns
    ///
    /// A vector of `Dimension` instances.
    pub fn create_dimensions(sizes: &[usize]) -> Vec<Dimension> {
        let mut dimensions = Vec::with_capacity(sizes.len());
        let mut stride = 1;
        
        for (i, &size) in sizes.iter().enumerate().rev() {
            dimensions.push(Dimension {
                index: i,
                size,
                stride,
            });
            
            stride *= size;
        }
        
        dimensions.reverse();
        dimensions
    }
} 