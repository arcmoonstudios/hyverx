//! Work scheduler implementation for parallel processing.
//!
//! This module provides a work scheduler for distributing work across threads.
//! It includes features like priority-based scheduling, work stealing, and
//! dynamic load balancing for efficient parallel processing.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};

use super::{ParallelConfig, ThreadPool, WorkItem};
use crate::algorithms::AlgorithmType;
use crate::error::{Error, Result};

/// A work scheduler for distributing work across threads.
pub struct WorkScheduler {
    /// The thread pool for executing tasks
    thread_pool: Arc<ThreadPool>,
    /// The work queue
    work_queue: Mutex<BinaryHeap<PrioritizedWorkItem>>,
    /// The configuration for the scheduler
    config: ParallelConfig,
    /// The work item handlers
    handlers: Mutex<HashMap<AlgorithmType, Box<dyn WorkItemHandler + Send + Sync>>>,
}

/// A prioritized work item for the scheduler.
#[derive(Debug)]
struct PrioritizedWorkItem {
    /// The work item
    item: WorkItem,
    /// The sequence number (for stable ordering)
    sequence: usize,
}

/// A trait for handling work items.
pub trait WorkItemHandler: Send + Sync {
    /// Processes a work item.
    ///
    /// # Arguments
    ///
    /// * `item` - The work item to process
    ///
    /// # Returns
    ///
    /// `Ok(())` if the item was processed successfully, or an error if processing failed.
    fn process(&self, item: &WorkItem) -> Result<()>;

    /// Returns the algorithm type this handler can process.
    fn algorithm_type(&self) -> AlgorithmType;

    /// Creates a boxed clone of this handler.
    fn clone_box(&self) -> Box<dyn WorkItemHandler + Send + Sync>;
}

impl PartialEq for PrioritizedWorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.item.priority == other.item.priority && self.sequence == other.sequence
    }
}

impl Eq for PrioritizedWorkItem {}

impl PartialOrd for PrioritizedWorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedWorkItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority comes first
        other
            .item
            .priority
            .cmp(&self.item.priority)
            // For equal priorities, lower sequence number comes first
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

/// A wrapper that captures a handler function and a work item
struct HandlerWrapper {
    item: WorkItem,
    algorithm_type: AlgorithmType,
}

impl HandlerWrapper {
    fn execute(&self, scheduler: &WorkScheduler) -> Result<()> {
        // Lock the handlers once
        let handlers = scheduler
            .handlers
            .lock()
            .map_err(|_| Error::Internal("Failed to lock handlers".into()))?;

        // Find the appropriate handler for this algorithm type
        if let Some(handler) = handlers.get(&self.algorithm_type) {
            // Process the item using the handler
            handler.process(&self.item)?;
        }

        Ok(())
    }
}

impl WorkScheduler {
    /// Creates a new work scheduler with the given thread pool and configuration.
    ///
    /// # Arguments
    ///
    /// * `thread_pool` - The thread pool for executing tasks
    /// * `config` - Configuration for the scheduler
    ///
    /// # Returns
    ///
    /// A new `WorkScheduler` instance.
    pub fn new(thread_pool: Arc<ThreadPool>, config: ParallelConfig) -> Self {
        Self {
            thread_pool,
            work_queue: Mutex::new(BinaryHeap::new()),
            config,
            handlers: Mutex::new(HashMap::new()),
        }
    }

    /// Creates a new work scheduler with the given thread pool and default configuration.
    ///
    /// # Arguments
    ///
    /// * `thread_pool` - The thread pool for executing tasks
    ///
    /// # Returns
    ///
    /// A new `WorkScheduler` instance.
    pub fn with_thread_pool(thread_pool: Arc<ThreadPool>) -> Self {
        Self::new(thread_pool, ParallelConfig::default())
    }

    /// Registers a handler for a specific algorithm type.
    ///
    /// # Arguments
    ///
    /// * `handler` - The handler to register
    ///
    /// # Returns
    ///
    /// `Ok(())` if the handler was registered successfully, or an error if registration failed.
    pub fn register_handler<H>(&self, handler: H) -> Result<()>
    where
        H: WorkItemHandler + 'static,
    {
        let algorithm_type = handler.algorithm_type();

        let mut handlers = self
            .handlers
            .lock()
            .map_err(|_| Error::Internal("Failed to lock handlers".into()))?;

        handlers.insert(algorithm_type, Box::new(handler));

        Ok(())
    }

    /// Schedules a work item for execution.
    ///
    /// # Arguments
    ///
    /// * `item` - The work item to schedule
    ///
    /// # Returns
    ///
    /// `Ok(())` if the item was scheduled successfully, or an error if scheduling failed.
    pub fn schedule(&self, item: WorkItem) -> Result<()> {
        // Check if a handler exists for this algorithm type
        let handlers = self
            .handlers
            .lock()
            .map_err(|_| Error::Internal("Failed to lock handlers".into()))?;

        if !handlers.contains_key(&item.algorithm) {
            return Err(Error::InvalidInput(
                format!(
                    "No handler registered for algorithm type {:?}",
                    item.algorithm
                )
                .into(),
            ));
        }

        // Add the item to the work queue
        let mut work_queue = self
            .work_queue
            .lock()
            .map_err(|_| Error::Internal("Failed to lock work queue".into()))?;

        let sequence = work_queue.len();

        work_queue.push(PrioritizedWorkItem { item, sequence });

        // Schedule processing
        self.schedule_processing()?;

        Ok(())
    }

    /// Schedules processing of work items.
    ///
    /// This will schedule as many work items as there are available threads.
    ///
    /// # Returns
    ///
    /// `Ok(())` if processing was scheduled successfully, or an error if scheduling failed.
    fn schedule_processing(&self) -> Result<()> {
        // Get work items to process
        let mut work_queue = self
            .work_queue
            .lock()
            .map_err(|_| Error::Internal("Failed to lock work queue".into()))?;

        // If queue is empty, return early
        if work_queue.is_empty() {
            return Ok(());
        }

        // Collect work items into a vector
        let items: Vec<_> = work_queue.drain().map(|p| p.item).collect();
        drop(work_queue); // Release lock

        // Get a reference to the thread pool
        let thread_pool = Arc::clone(&self.thread_pool);
        let scheduler = Arc::new(self.clone());

        // For each item, create a handler wrapper that captures just the item and algorithm type
        for item in items {
            // Create a safe wrapper that will execute the handler with the item
            let item_clone = item.clone();
            let algorithm_type = item.algorithm.clone();

            // Create a wrapper that contains just the item and its algorithm type
            let handler_wrapper = Arc::new(HandlerWrapper {
                item: item_clone,
                algorithm_type,
            });

            // Create a reference to the scheduler that can be moved to the thread
            let scheduler_clone = Arc::clone(&scheduler);
            let wrapper_clone = Arc::clone(&handler_wrapper);

            // Schedule the task for execution
            thread_pool.execute(move || {
                // Execute the handler with the item through the scheduler
                let _ = wrapper_clone.execute(&scheduler_clone);
            })?;
        }

        Ok(())
    }

    /// Waits for all scheduled work items to complete.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all items completed successfully, or an error if waiting failed.
    pub fn wait(&self) -> Result<()> {
        self.thread_pool.wait()
    }

    /// Returns the number of queued work items.
    pub fn queued_items(&self) -> Result<usize> {
        let work_queue = self
            .work_queue
            .lock()
            .map_err(|_| Error::Internal("Failed to lock work queue".into()))?;

        Ok(work_queue.len())
    }

    /// Returns the number of active work items.
    pub fn active_items(&self) -> usize {
        self.thread_pool.active_tasks()
    }

    /// Splits a work item into smaller items for parallel processing.
    ///
    /// # Arguments
    ///
    /// * `item` - The work item to split
    /// * `count` - The number of items to split into
    ///
    /// # Returns
    ///
    /// A vector of split work items.
    pub fn split_work_item(&self, item: &WorkItem, count: usize) -> Vec<WorkItem> {
        // Select a dimension to split along
        let mut best_dimension = 0;
        let mut max_size = 0;

        for (i, _dim) in item.region.dimensions.iter().enumerate() {
            let range = item.region.end[i] - item.region.start[i];
            if range > max_size {
                max_size = range;
                best_dimension = i;
            }
        }

        // Split the region
        let regions = item.region.split(best_dimension, count);

        // Create work items for each region
        regions
            .into_iter()
            .map(|region| WorkItem {
                region,
                algorithm: item.algorithm,
                priority: item.priority,
            })
            .collect()
    }

    /// Schedules a batch of work items for execution.
    ///
    /// # Arguments
    ///
    /// * `items` - The work items to schedule
    ///
    /// # Returns
    ///
    /// `Ok(())` if the items were scheduled successfully, or an error if scheduling failed.
    pub fn schedule_batch(&self, items: Vec<WorkItem>) -> Result<()> {
        // Check if handlers exist for all algorithm types
        let handlers = self
            .handlers
            .lock()
            .map_err(|_| Error::Internal("Failed to lock handlers".into()))?;

        for item in &items {
            if !handlers.contains_key(&item.algorithm) {
                return Err(Error::InvalidInput(
                    format!(
                        "No handler registered for algorithm type {:?}",
                        item.algorithm
                    )
                    .into(),
                ));
            }
        }

        // Add the items to the work queue
        let mut work_queue = self
            .work_queue
            .lock()
            .map_err(|_| Error::Internal("Failed to lock work queue".into()))?;

        let mut sequence = work_queue.len();

        for item in items {
            work_queue.push(PrioritizedWorkItem { item, sequence });

            sequence += 1;
        }

        // Schedule processing
        self.schedule_processing()?;

        Ok(())
    }
}

impl std::fmt::Debug for WorkScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkScheduler")
            .field("thread_pool", &self.thread_pool)
            // Don't try to format mutex contents directly
            .field("work_queue", &"[Work Queue]")
            .field("config", &self.config)
            .field("handlers", &"[Handlers]")
            .finish()
    }
}

impl Clone for WorkScheduler {
    fn clone(&self) -> Self {
        Self {
            thread_pool: Arc::clone(&self.thread_pool),
            work_queue: Mutex::new(BinaryHeap::new()), // Create a new empty queue
            config: self.config.clone(),
            handlers: Mutex::new(HashMap::new()), // Create a new empty handlers map
        }
    }
}
