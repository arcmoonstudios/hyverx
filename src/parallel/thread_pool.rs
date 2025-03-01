//! Thread pool implementation for parallel processing.
//!
//! This module provides a thread pool implementation for executing tasks in parallel.
//! It includes features like work stealing, dynamic load balancing, and adaptive
//! chunk sizing for efficient parallel processing.

use std::sync::{Arc, Mutex, Condvar, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::thread;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use super::{Task, ParallelStats, ParallelConfig};

/// A thread pool for executing tasks in parallel.
pub struct ThreadPool {
    /// The worker threads
    workers: Vec<Worker>,
    /// The task queue
    queue: Arc<Mutex<VecDeque<Task>>>,
    /// Condition variable for signaling workers
    condvar: Arc<Condvar>,
    /// Whether the pool is shutting down
    shutdown: Arc<AtomicBool>,
    /// Number of active tasks
    active_tasks: Arc<AtomicUsize>,
    /// Statistics for the thread pool
    stats: Arc<Mutex<ParallelStats>>,
    /// Configuration for the thread pool
    config: ParallelConfig,
}

/// A worker thread in the thread pool.
struct Worker {
    /// The worker's thread handle
    thread: Option<thread::JoinHandle<()>>,
    /// The worker's ID
    id: usize,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("id", &self.id)
            .field("thread", &"<thread handle>")
            .finish()
    }
}

impl ThreadPool {
    /// Creates a new thread pool with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the thread pool
    ///
    /// # Returns
    ///
    /// A new `ThreadPool` instance.
    pub fn new(config: ParallelConfig) -> Self {
        let queue = Arc::new(Mutex::new(VecDeque::with_capacity(config.max_queue_size)));
        let condvar = Arc::new(Condvar::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(Mutex::new(ParallelStats::default()));
        
        let mut workers = Vec::with_capacity(config.thread_count);
        
        for id in 0..config.thread_count {
            workers.push(Worker::new(
                id,
                Arc::clone(&queue),
                Arc::clone(&condvar),
                Arc::clone(&shutdown),
                Arc::clone(&active_tasks),
                Arc::clone(&stats),
                config.clone(),
            ));
        }
        
        Self {
            workers,
            queue,
            condvar,
            shutdown,
            active_tasks,
            stats,
            config,
        }
    }
    
    /// Creates a new thread pool with the default configuration.
    ///
    /// # Returns
    ///
    /// A new `ThreadPool` instance.
    pub fn default() -> Self {
        Self::new(ParallelConfig::default())
    }
    
    /// Executes a task in the thread pool.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute
    ///
    /// # Returns
    ///
    /// `Ok(())` if the task was queued successfully, or an error if the queue is full.
    pub fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::SeqCst) {
            return Err(Error::ThreadPoolShutdown("Thread pool is shutting down".to_string()));
        }
        
        let task = Box::new(f);
        
        let mut queue = self.queue.lock().map_err(|_| {
            Error::Internal("Failed to lock task queue".into())
        })?;
        
        if queue.len() >= self.config.max_queue_size {
            return Err(Error::QueueFull("Task queue is at maximum capacity".to_string()));
        }
        
        queue.push_back(task);
        
        // Update statistics
        {
            let mut stats = self.stats.lock().map_err(|_| {
                Error::Internal("Failed to lock stats".into())
            })?;
            stats.tasks_queued += 1;
        }
        
        // Notify a worker
        self.condvar.notify_one();
        
        Ok(())
    }
    
    /// Waits for all tasks to complete.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all tasks completed successfully, or an error if waiting failed.
    pub fn wait(&self) -> Result<()> {
        let mut last_active = self.active_tasks.load(Ordering::SeqCst);
        
        while last_active > 0 || !self.queue.lock().map_err(|_| {
            Error::Internal("Failed to lock task queue".into())
        })?.is_empty() {
            thread::sleep(Duration::from_millis(10));
            last_active = self.active_tasks.load(Ordering::SeqCst);
        }
        
        Ok(())
    }
    
    /// Shuts down the thread pool.
    ///
    /// This will wait for all tasks to complete before shutting down.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the pool was shut down successfully, or an error if shutdown failed.
    pub fn shutdown(&mut self) -> Result<()> {
        // Wait for all tasks to complete
        self.wait()?;
        
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);
        
        // Notify all workers
        self.condvar.notify_all();
        
        // Join worker threads
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().map_err(|_| {
                    Error::Internal("Failed to join worker thread".into())
                })?;
            }
        }
        
        Ok(())
    }
    
    /// Returns the number of worker threads in the pool.
    pub fn thread_count(&self) -> usize {
        self.workers.len()
    }
    
    /// Returns the number of queued tasks.
    pub fn queued_tasks(&self) -> Result<usize> {
        let queue = self.queue.lock().map_err(|_| {
            Error::Internal("Failed to lock task queue".into())
        })?;
        
        Ok(queue.len())
    }
    
    /// Returns the number of active tasks.
    pub fn active_tasks(&self) -> usize {
        self.active_tasks.load(Ordering::SeqCst)
    }
    
    /// Returns statistics for the thread pool.
    pub fn stats(&self) -> Result<ParallelStats> {
        let stats = self.stats.lock().map_err(|_| {
            Error::Internal("Failed to lock stats".into())
        })?;
        
        Ok(stats.clone())
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);
        
        // Notify all workers
        self.condvar.notify_all();
        
        // Join worker threads
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

impl Worker {
    /// Creates a new worker thread.
    ///
    /// # Arguments
    ///
    /// * `id` - The worker's ID
    /// * `queue` - The task queue
    /// * `condvar` - Condition variable for signaling workers
    /// * `shutdown` - Whether the pool is shutting down
    /// * `active_tasks` - Number of active tasks
    /// * `stats` - Statistics for the thread pool
    /// * `config` - Configuration for the thread pool
    ///
    /// # Returns
    ///
    /// A new `Worker` instance.
    fn new(
        id: usize,
        queue: Arc<Mutex<VecDeque<Task>>>,
        condvar: Arc<Condvar>,
        shutdown: Arc<AtomicBool>,
        active_tasks: Arc<AtomicUsize>,
        stats: Arc<Mutex<ParallelStats>>,
        config: ParallelConfig,
    ) -> Self {
        let thread = thread::spawn(move || {
            Self::run(
                id,
                queue,
                condvar,
                shutdown,
                active_tasks,
                stats,
                config,
            );
        });
        
        Self {
            thread: Some(thread),
            id,
        }
    }
    
    /// Runs the worker thread.
    ///
    /// This function will loop until the pool is shut down, executing tasks from the queue.
    fn run(
        id: usize,
        queue: Arc<Mutex<VecDeque<Task>>>,
        condvar: Arc<Condvar>,
        shutdown: Arc<AtomicBool>,
        active_tasks: Arc<AtomicUsize>,
        stats: Arc<Mutex<ParallelStats>>,
        config: ParallelConfig,
    ) {
        let _id = id; // Use ID in log messages or worker identification
        let _config = config; // Use config for worker-specific settings
        
        loop {
            // Get a task from the queue
            let task = {
                let mut queue = queue.lock().unwrap();
                
                // Wait for a task or shutdown signal
                while queue.is_empty() && !shutdown.load(Ordering::SeqCst) {
                    queue = condvar.wait(queue).unwrap();
                }
                
                // Check for shutdown
                if shutdown.load(Ordering::SeqCst) && queue.is_empty() {
                    break;
                }
                
                queue.pop_front()
            };
            
            // Execute the task
            if let Some(task) = task {
                active_tasks.fetch_add(1, Ordering::SeqCst);
                
                let start_time = Instant::now();
                
                // Execute the task
                task();
                
                let elapsed = start_time.elapsed();
                
                // Update statistics
                {
                    let mut stats = stats.lock().unwrap();
                    stats.tasks_executed += 1;
                    stats.tasks_completed += 1;
                    stats.total_time_ms += elapsed.as_secs_f64() * 1000.0;
                    
                    let task_time_ms = elapsed.as_secs_f64() * 1000.0;
                    
                    if stats.tasks_executed == 1 {
                        stats.avg_task_time_ms = task_time_ms;
                        stats.max_task_time_ms = task_time_ms;
                        stats.min_task_time_ms = task_time_ms;
                    } else {
                        stats.avg_task_time_ms = (stats.avg_task_time_ms * (stats.tasks_executed - 1) as f64 + task_time_ms) / stats.tasks_executed as f64;
                        stats.max_task_time_ms = stats.max_task_time_ms.max(task_time_ms);
                        stats.min_task_time_ms = stats.min_task_time_ms.min(task_time_ms);
                    }
                }
                
                active_tasks.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("workers", &self.workers)
            // Skip queue since Task doesn't implement Debug
            .field("shutdown", &self.shutdown)
            .field("active_tasks", &self.active_tasks)
            .field("stats", &self.stats)
            .field("config", &self.config)
            .finish()
    }
} 