//! # Parallel Processing Utilities
//! 
//! This module provides utilities for parallel processing and multi-dimensional
//! error correction. It includes thread pool management, work scheduling, and
//! dimension mapping for efficient parallel processing of error correction tasks.
//!
//! ## Components
//!
//! - `ThreadPool`: A thread pool for executing tasks in parallel
//! - `WorkScheduler`: A scheduler for distributing work across threads
//! - `DimensionMapper`: A utility for mapping multi-dimensional data to threads
//! - `AlgorithmAllocator`: A utility for allocating algorithms to error patterns

use crate::algorithms::AlgorithmType;
use crate::error::{Error, Result};

// Submodules
mod thread_pool;
mod work_scheduler;
mod dimension_mapper;
mod algorithm_allocator;

// Public exports
pub use thread_pool::ThreadPool;
pub use work_scheduler::WorkScheduler;
pub use dimension_mapper::DimensionMapper;
pub use algorithm_allocator::AlgorithmAllocator;

/// A task to be executed by the thread pool.
pub type Task = Box<dyn FnOnce() + Send + 'static>;

/// A dimension in multi-dimensional data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dimension {
    /// The index of the dimension
    pub index: usize,
    /// The size of the dimension
    pub size: usize,
    /// The stride of the dimension
    pub stride: usize,
}

/// A region in multi-dimensional data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// The dimensions of the region
    pub dimensions: Vec<Dimension>,
    /// The start indices for each dimension
    pub start: Vec<usize>,
    /// The end indices for each dimension
    pub end: Vec<usize>,
}

impl Region {
    /// Creates a new region with the given dimensions.
    pub fn new(dimensions: Vec<Dimension>) -> Self {
        let start = vec![0; dimensions.len()];
        let end = dimensions.iter().map(|d| d.size).collect();
        
        Self {
            dimensions,
            start,
            end,
        }
    }
    
    /// Creates a new region with the given dimensions, start, and end.
    pub fn with_bounds(dimensions: Vec<Dimension>, start: Vec<usize>, end: Vec<usize>) -> Self {
        Self {
            dimensions,
            start,
            end,
        }
    }
    
    /// Returns the total number of elements in the region.
    pub fn size(&self) -> usize {
        self.dimensions.iter()
            .zip(self.start.iter().zip(self.end.iter()))
            .map(|(_d, (s, e))| e - s)
            .product()
    }
    
    /// Splits the region into subregions along the given dimension.
    pub fn split(&self, dimension_index: usize, count: usize) -> Vec<Self> {
        if dimension_index >= self.dimensions.len() {
            return vec![self.clone()];
        }
        
        let dim_size = self.end[dimension_index] - self.start[dimension_index];
        let chunk_size = (dim_size + count - 1) / count; // Ceiling division
        
        let mut regions = Vec::with_capacity(count);
        
        for i in 0..count {
            let mut start = self.start.clone();
            let mut end = self.end.clone();
            
            start[dimension_index] = self.start[dimension_index] + i * chunk_size;
            end[dimension_index] = (self.start[dimension_index] + (i + 1) * chunk_size).min(self.end[dimension_index]);
            
            // Skip empty regions
            if start[dimension_index] >= end[dimension_index] {
                continue;
            }
            
            regions.push(Self::with_bounds(
                self.dimensions.clone(),
                start,
                end,
            ));
        }
        
        regions
    }
    
    /// Returns an iterator over the indices in the region.
    pub fn indices(&self) -> RegionIndicesIterator {
        RegionIndicesIterator {
            region: self.clone(),
            current: self.start.clone(),
            done: false,
        }
    }
}

/// An iterator over the indices in a region.
#[derive(Debug)]
pub struct RegionIndicesIterator {
    /// The region being iterated over
    region: Region,
    /// The current indices
    current: Vec<usize>,
    /// Whether the iteration is done
    done: bool,
}

impl Iterator for RegionIndicesIterator {
    type Item = Vec<usize>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        
        let result = self.current.clone();
        
        // Increment indices
        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            
            if self.current[i] < self.region.end[i] {
                return Some(result);
            }
            
            self.current[i] = self.region.start[i];
        }
        
        // If we get here, we've wrapped around all dimensions
        self.done = true;
        Some(result)
    }
}

/// A work item to be processed by the thread pool.
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// The region to process
    pub region: Region,
    /// The algorithm to use
    pub algorithm: AlgorithmType,
    /// The priority of the work item
    pub priority: usize,
}

/// Statistics for parallel processing.
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    /// Number of tasks executed
    pub tasks_executed: usize,
    /// Number of tasks queued
    pub tasks_queued: usize,
    /// Number of tasks completed
    pub tasks_completed: usize,
    /// Number of tasks failed
    pub tasks_failed: usize,
    /// Total processing time (ms)
    pub total_time_ms: f64,
    /// Average task time (ms)
    pub avg_task_time_ms: f64,
    /// Maximum task time (ms)
    pub max_task_time_ms: f64,
    /// Minimum task time (ms)
    pub min_task_time_ms: f64,
    /// Thread utilization (0.0-1.0)
    pub thread_utilization: f64,
}

/// Configuration for parallel processing.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use
    pub thread_count: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Work stealing enabled
    pub work_stealing: bool,
    /// Dynamic load balancing enabled
    pub dynamic_load_balancing: bool,
    /// Minimum chunk size for splitting
    pub min_chunk_size: usize,
    /// Maximum chunk size for splitting
    pub max_chunk_size: usize,
    /// Adaptive chunk sizing enabled
    pub adaptive_chunk_sizing: bool,
    /// Thread priority (1-10)
    pub thread_priority: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            thread_count: num_cpus::get(),
            max_queue_size: 1000,
            work_stealing: true,
            dynamic_load_balancing: true,
            min_chunk_size: 1024,
            max_chunk_size: 1024 * 1024,
            adaptive_chunk_sizing: true,
            thread_priority: 5,
        }
    }
}

/// A utility for mapping data across multiple dimensions.
pub fn map_dimensions(data: &[u8], dimensions: &[usize]) -> Result<Vec<Vec<u8>>> {
    if dimensions.is_empty() {
        return Err(Error::InvalidInput("Dimensions cannot be empty".into()));
    }
    
    let total_size: usize = dimensions.iter().product();
    if total_size != data.len() {
        return Err(Error::InvalidInput(
            format!(
                "Data size ({}) does not match dimensions product ({})",
                data.len(),
                total_size
            )
            .into(),
        ));
    }
    
    let mut result = Vec::with_capacity(dimensions[0]);
    let slice_size = total_size / dimensions[0];
    
    for i in 0..dimensions[0] {
        let start = i * slice_size;
        let end = start + slice_size;
        result.push(data[start..end].to_vec());
    }
    
    Ok(result)
}

/// A utility for splitting data into chunks for parallel processing.
pub fn split_for_parallel(data: &[u8], chunk_count: usize) -> Vec<Vec<u8>> {
    let chunk_size = (data.len() + chunk_count - 1) / chunk_count; // Ceiling division
    
    let mut chunks = Vec::with_capacity(chunk_count);
    
    for i in 0..chunk_count {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(data.len());
        
        if start >= end {
            break;
        }
        
        chunks.push(data[start..end].to_vec());
    }
    
    chunks
}

/// A utility for merging results from parallel processing.
pub fn merge_parallel_results(results: Vec<Vec<u8>>) -> Vec<u8> {
    let total_size: usize = results.iter().map(|r| r.len()).sum();
    let mut merged = Vec::with_capacity(total_size);
    
    for result in results {
        merged.extend_from_slice(&result);
    }
    
    merged
}

/// A utility for calculating the optimal chunk size for parallel processing.
pub fn calculate_optimal_chunk_size(data_size: usize, thread_count: usize) -> usize {
    let min_chunk_size = 1024;
    let max_chunk_size = 1024 * 1024;
    
    let chunk_size = data_size / thread_count;
    
    chunk_size.max(min_chunk_size).min(max_chunk_size)
}

/// A utility for calculating the optimal thread count for parallel processing.
pub fn calculate_optimal_thread_count(data_size: usize) -> usize {
    let available_threads = num_cpus::get();
    let min_data_per_thread = 1024;
    
    let thread_count = data_size / min_data_per_thread;
    
    thread_count.min(available_threads).max(1)
}
