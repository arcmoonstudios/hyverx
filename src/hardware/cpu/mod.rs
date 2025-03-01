//! Hardware acceleration implementations using CPU-specific optimizations.
//!
//! This module provides CPU-optimized implementations for hardware acceleration
//! tasks, leveraging AVX2 vector extensions and OpenMP multi-threading for
//! performance on modern CPUs.
//!
//! It includes:
//!
//! - AVX2-optimized matrix and vector operations
//! - OpenMP parallelized algorithms
//! - CPU-optimized Galois field operations for error correction
//! - Neural network primitives optimized for multi-core CPUs

use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::galois::GaloisField;
use crate::hardware::AcceleratorType;
use crate::hardware::HardwareAccelerator;
use crate::hardware::HardwareCapabilities;
use crate::hardware::HardwareStatistics;
use crate::hardware::TensorOperation;
use crate::hardware::ElementWiseOp;

/// FFI bindings for AVX2 kernels - provides low-level access to AVX2 SIMD instructions
/// for hardware acceleration on CPUs supporting Advanced Vector Extensions 2.
#[allow(non_camel_case_types)]
pub mod avx2_ffi {
    use std::os::raw::{c_float, c_int, c_ushort};
    
    /// Element-wise operations supported by AVX2 kernels
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub enum ElementWiseOperation {
        /// Rectified Linear Unit activation function: f(x) = max(0, x)
        RELU,
        /// Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
        SIGMOID,
        /// Hyperbolic tangent activation function: f(x) = tanh(x)
        TANH,
        /// Addition operation: f(x) = x + c
        ADD,
        /// Multiplication operation: f(x) = x * c
        MULTIPLY,
    }
    
    extern "C" {
        /// Checks if AVX2 instruction set is supported on the current CPU
        pub fn is_avx2_supported() -> bool;
        
        /// Performs matrix multiplication using AVX2 instructions
        /// 
        /// Calculates C = A * B where A is (m x k) and B is (k x n)
        pub fn matrix_multiply(
            a: *const c_float,
            b: *const c_float,
            c: *mut c_float,
            m: c_int,
            n: c_int,
            k: c_int,
        );
        
        /// Performs element-wise operations using AVX2 instructions
        /// 
        /// Applies the specified operation to each element of the input array
        pub fn element_wise_operation(
            input: *const c_float,
            output: *mut c_float,
            size: c_int,
            op: ElementWiseOperation,
            constant: c_float,
        );
        
        /// Performs 2D convolution using AVX2 instructions
        /// 
        /// Applies a convolution kernel to an input tensor
        pub fn convolution(
            input: *const c_float,
            kernel: *const c_float,
            output: *mut c_float,
            batch_size: c_int,
            input_height: c_int,
            input_width: c_int,
            input_channels: c_int,
            kernel_height: c_int,
            kernel_width: c_int,
            output_channels: c_int,
            output_height: c_int,
            output_width: c_int,
            stride_h: c_int,
            stride_w: c_int,
            padding_h: c_int,
            padding_w: c_int,
        );
        
        /// Performs Galois field multiplication using AVX2 instructions
        /// 
        /// Multiplies two arrays of field elements element-wise
        pub fn gf_multiply(
            a: *const c_ushort,
            b: *const c_ushort,
            result: *mut c_ushort,
            size: c_int,
            exp_table: *const c_ushort,
            log_table: *const c_ushort,
            field_size: c_int,
        );
        
        /// Performs Galois field addition using AVX2 instructions
        /// 
        /// Adds two arrays of field elements element-wise (XOR operation)
        pub fn gf_add(
            a: *const c_ushort,
            b: *const c_ushort,
            result: *mut c_ushort,
            size: c_int,
        );
        
        /// Calculates syndromes for error detection using AVX2 instructions
        /// 
        /// Computes syndrome values for the given data array
        pub fn calculate_syndromes(
            data: *const c_ushort,
            data_length: c_int,
            syndromes: *mut c_ushort,
            syndrome_count: c_int,
            exp_table: *const c_ushort,
            log_table: *const c_ushort,
            field_size: c_int,
        );
        
        /// Evaluates polynomials at multiple points in batch using AVX2 instructions
        /// 
        /// Efficiently evaluates multiple polynomials at multiple points in parallel
        pub fn polynomial_eval_batch(
            polys: *const c_ushort,
            poly_lengths: *const c_int,
            max_poly_len: c_int,
            n_polys: c_int,
            points: *const c_ushort,
            n_points: c_int,
            results: *mut c_ushort,
            log_table: *const c_ushort,
            exp_table: *const c_ushort,
            field_size: c_int,
        );
    }
}

/// FFI bindings for OpenMP kernels - provides low-level access to OpenMP parallelization
/// for hardware acceleration using multi-threading on CPUs.
#[allow(non_camel_case_types)]
pub mod openmp_ffi {
    use std::os::raw::{c_float, c_int, c_ushort};
    
    /// Element-wise operations supported by OpenMP kernels
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub enum ElementWiseOperation {
        /// Rectified Linear Unit activation function: f(x) = max(0, x)
        RELU,
        /// Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
        SIGMOID,
        /// Hyperbolic tangent activation function: f(x) = tanh(x)
        TANH,
        /// Addition operation: f(x) = x + c
        ADD,
        /// Multiplication operation: f(x) = x * c
        MULTIPLY,
    }
    
    extern "C" {
        /// Performs matrix multiplication using OpenMP parallelization
        /// 
        /// Calculates C = A * B where A is (m x k) and B is (k x n)
        pub fn openmp_matrix_multiply(
            a: *const c_float,
            b: *const c_float,
            c: *mut c_float,
            m: c_int,
            n: c_int,
            k: c_int,
            num_threads: c_int,
        );
        
        /// Performs element-wise operations using OpenMP parallelization
        /// 
        /// Applies the specified operation to each element of the input array
        pub fn openmp_element_wise_operation(
            input: *const c_float,
            output: *mut c_float,
            size: c_int,
            op: ElementWiseOperation,
            constant: c_float,
            num_threads: c_int,
        );
        
        /// Performs 2D convolution using OpenMP parallelization
        /// 
        /// Applies a convolution kernel to an input tensor
        pub fn openmp_convolution(
            input: *const c_float,
            kernel: *const c_float,
            output: *mut c_float,
            batch_size: c_int,
            input_height: c_int,
            input_width: c_int,
            input_channels: c_int,
            kernel_height: c_int,
            kernel_width: c_int,
            output_channels: c_int,
            output_height: c_int,
            output_width: c_int,
            stride_h: c_int,
            stride_w: c_int,
            padding_h: c_int,
            padding_w: c_int,
            num_threads: c_int,
        );
        
        /// Performs Galois field multiplication using OpenMP parallelization
        /// 
        /// Multiplies two arrays of field elements element-wise
        pub fn openmp_gf_multiply(
            a: *const c_ushort,
            b: *const c_ushort,
            result: *mut c_ushort,
            size: c_int,
            exp_table: *const c_ushort,
            log_table: *const c_ushort,
            field_size: c_int,
            num_threads: c_int,
        );
        
        /// Performs Galois field addition using OpenMP parallelization
        /// 
        /// Adds two arrays of field elements element-wise (XOR operation)
        pub fn openmp_gf_add(
            a: *const c_ushort,
            b: *const c_ushort,
            result: *mut c_ushort,
            size: c_int,
            num_threads: c_int,
        );
        
        /// Calculates syndromes for error detection using OpenMP parallelization
        /// 
        /// Computes syndrome values for the given data array
        pub fn openmp_calculate_syndromes(
            data: *const c_ushort,
            data_length: c_int,
            syndromes: *mut c_ushort,
            syndrome_count: c_int,
            exp_table: *const c_ushort,
            log_table: *const c_ushort,
            field_size: c_int,
            num_threads: c_int,
        );
        
        /// Evaluates polynomials at multiple points in batch using OpenMP parallelization
        /// 
        /// Efficiently evaluates multiple polynomials at multiple points in parallel
        pub fn openmp_polynomial_eval_batch(
            polys: *const c_ushort,
            poly_lengths: *const c_int,
            max_poly_len: c_int,
            n_polys: c_int,
            points: *const c_ushort,
            n_points: c_int,
            results: *mut c_ushort,
            log_table: *const c_ushort,
            exp_table: *const c_ushort,
            field_size: c_int,
            num_threads: c_int,
        );
        
        /// Returns the optimal number of threads for OpenMP operations
        /// 
        /// Uses system information to determine the best thread count for performance
        pub fn get_optimal_thread_count() -> c_int;
    }
}

/// CPU accelerator using AVX2 SIMD instructions.
/// 
/// This implementation leverages Advanced Vector Extensions 2 (AVX2) for SIMD parallelism,
/// enabling efficient execution of matrix and vector operations on CPU architectures
/// that support these extensions.
#[derive(Debug)]
pub struct CPUAccelerator {
    /// Galois field for arithmetic operations
    galois_field: Arc<GaloisField>,
    /// Whether AVX2 is available on the current CPU
    avx2_available: bool,
    /// Statistics for hardware operations
    stats: RwLock<HardwareStatistics>,
}

impl CPUAccelerator {
    /// Creates a new AVX2 accelerator.
    ///
    /// # Arguments
    ///
    /// * `galois_field` - Galois field for arithmetic operations
    ///
    /// # Returns
    ///
    /// A new AVX2 accelerator
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub fn new(galois_field: Arc<GaloisField>) -> Result<Self> {
        // Check AVX2 support
        let avx2_available = unsafe { avx2_ffi::is_avx2_supported() };
        
        Ok(Self {
            galois_field,
            avx2_available,
            stats: RwLock::new(HardwareStatistics::default()),
        })
    }
    
    /// Checks if AVX2 accelerator is available on the current system.
    pub fn is_available() -> bool {
        unsafe { avx2_ffi::is_avx2_supported() }
    }
    
    /// Matrix multiplication using AVX2 instructions.
    /// 
    /// Performs optimized matrix multiplication C = A * B using AVX2 SIMD instructions.
    ///
    /// # Arguments
    ///
    /// * `a` - First matrix (m x k)
    /// * `b` - Second matrix (k x n)
    /// * `c` - Result matrix (m x n)
    /// * `dims` - Dimensions (m, k, n)
    ///
    /// # Returns
    ///
    /// Result indicating success or an error
    fn matrix_multiply_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        c: Arc<parking_lot::Mutex<Vec<f32>>>,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        let (m, k, n) = dims;
        
        // Create output buffer
        let mut c_data = vec![0.0f32; m * n];
        
        // Call AVX2 matrix multiplication
        unsafe {
            avx2_ffi::matrix_multiply(
                a.as_ptr(),
                b.as_ptr(),
                c_data.as_mut_ptr(),
                m as i32,
                n as i32,
                k as i32,
            );
        }
        
        // Update output
        let mut c_guard = c.lock();
        *c_guard = c_data;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(())
    }
    
    /// Element-wise operation using AVX2 instructions.
    /// 
    /// Applies an element-wise operation to a tensor using AVX2 SIMD instructions.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `output` - Output tensor
    /// * `op` - Element-wise operation to apply
    ///
    /// # Returns
    ///
    /// Result indicating success or an error
    fn element_wise_avx2(
        &self,
        input: &[f32],
        output: Arc<parking_lot::Mutex<Vec<f32>>>,
        op: ElementWiseOp,
    ) -> Result<()> {
        // Convert ElementWiseOp to AVX2 ElementWiseOperation
        let avx2_op = match op {
            ElementWiseOp::ReLU => avx2_ffi::ElementWiseOperation::RELU,
            ElementWiseOp::Sigmoid => avx2_ffi::ElementWiseOperation::SIGMOID,
            ElementWiseOp::Tanh => avx2_ffi::ElementWiseOperation::TANH,
            ElementWiseOp::Add(_val) => avx2_ffi::ElementWiseOperation::ADD,
            ElementWiseOp::Multiply(_val) => avx2_ffi::ElementWiseOperation::MULTIPLY,
        };
        
        // Get constant value for Add/Multiply operations
        let constant = match op {
            ElementWiseOp::Add(val) => val,
            ElementWiseOp::Multiply(val) => val,
            _ => 0.0,
        };
        
        // Create output buffer
        let mut output_data = vec![0.0f32; input.len()];
        
        // Call AVX2 element-wise operation
        unsafe {
            avx2_ffi::element_wise_operation(
                input.as_ptr(),
                output_data.as_mut_ptr(),
                input.len() as i32,
                avx2_op,
                constant,
            );
        }
        
        // Update output
        let mut output_guard = output.lock();
        *output_guard = output_data;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(())
    }
    
    /// Convolution using AVX2 instructions.
    /// 
    /// Performs optimized 2D convolution using AVX2 SIMD instructions.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `kernel` - Convolution kernel
    /// * `output` - Output tensor
    /// * `input_dims` - Input dimensions (batch_size, height, width, channels)
    /// * `kernel_dims` - Kernel dimensions (height, width, in_channels, out_channels)
    /// * `stride` - Stride (height, width)
    /// * `padding` - Padding (height, width)
    ///
    /// # Returns
    ///
    /// Result indicating success or an error
    fn convolution_avx2(
        &self,
        input: &[f32],
        kernel: &[f32],
        output: Arc<parking_lot::Mutex<Vec<f32>>>,
        input_dims: (usize, usize, usize, usize),
        kernel_dims: (usize, usize, usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<()> {
        // Extract dimensions
        let (batch_size, input_height, input_width, input_channels) = input_dims;
        let (kernel_height, kernel_width, kernel_in_channels, kernel_out_channels) = kernel_dims;
        let (stride_h, stride_w) = stride;
        let (padding_h, padding_w) = padding;
        
        // Check dimensions
        if input_channels != kernel_in_channels {
            return Err(Error::InvalidInput(format!(
                "Input channels ({}) must match kernel input channels ({})",
                input_channels, kernel_in_channels
            )));
        }
        
        // Calculate output dimensions
        let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
        let output_size = batch_size * output_height * output_width * kernel_out_channels;
        
        // Create output buffer
        let mut output_data = vec![0.0f32; output_size];
        
        // Call AVX2 convolution
        unsafe {
            avx2_ffi::convolution(
                input.as_ptr(),
                kernel.as_ptr(),
                output_data.as_mut_ptr(),
                batch_size as i32,
                input_height as i32,
                input_width as i32,
                input_channels as i32,
                kernel_height as i32,
                kernel_width as i32,
                kernel_out_channels as i32,
                output_height as i32,
                output_width as i32,
                stride_h as i32,
                stride_w as i32,
                padding_h as i32,
                padding_w as i32,
            );
        }
        
        // Update output
        let mut output_guard = output.lock();
        *output_guard = output_data;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(())
    }
}

impl HardwareAccelerator for CPUAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::Cpu
    }
    
    fn is_available(&self) -> bool {
        self.avx2_available
    }
    
    fn capabilities(&self) -> HardwareCapabilities {
        HardwareCapabilities {
            avx2_available: self.avx2_available,
            openmp_available: false,
            cuda_available: false,
            cuda_device_count: 0,
            opencl_available: false,
            opencl_device_count: 0,
            processor_count: num_cpus::get(),
            available_memory: available_memory(),
            hamming_available: true,
            bch_available: true,
            reed_muller_available: true,
            turbo_available: false,
            convolutional_available: true,
            fountain_available: false,
        }
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the AVX2-specific tables directory
        let avx2_path = path.join("avx2");
        std::fs::create_dir_all(&avx2_path)?;
        
        // Generate Galois field lookup tables
        self.galois_field().generate_lookup_tables()?;
        
        Ok(())
    }
    
    fn calculate_syndromes(&self, data: &[u8], syndrome_count: usize) -> Result<Vec<u16>> {
        // Convert data to u16
        let data_u16: Vec<u16> = data.iter().map(|&x| x as u16).collect();
        
        // Get Galois field tables
        let exp_table = self.galois_field.get_exp_table()?;
        let log_table = self.galois_field.get_log_table()?;
        
        // Create output buffer
        let mut syndromes = vec![0u16; syndrome_count];
        
        // Call AVX2 syndrome calculation
        unsafe {
            avx2_ffi::calculate_syndromes(
                data_u16.as_ptr(),
                data_u16.len() as i32,
                syndromes.as_mut_ptr(),
                syndrome_count as i32,
                exp_table.as_ptr(),
                log_table.as_ptr(),
                self.galois_field.element_count() as i32,
            );
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(syndromes)
    }
    
    fn multiply_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput("Input arrays must have the same length".into()));
        }
        
        // Get Galois field tables
        let exp_table = self.galois_field.get_exp_table()?;
        let log_table = self.galois_field.get_log_table()?;
        
        // Create output buffer
        let mut result = vec![0u16; a.len()];
        
        // Call AVX2 Galois field multiplication
        unsafe {
            avx2_ffi::gf_multiply(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                a.len() as i32,
                exp_table.as_ptr(),
                log_table.as_ptr(),
                self.galois_field.element_count() as i32,
            );
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(result)
    }
    
    fn add_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput("Input arrays must have the same length".into()));
        }
        
        // Create output buffer
        let mut result = vec![0u16; a.len()];
        
        // Call AVX2 Galois field addition
        unsafe {
            avx2_ffi::gf_add(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                a.len() as i32,
            );
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(result)
    }
    
    fn polynomial_eval_batch(&self, polys: &[Vec<u16>], points: &[u16]) -> Result<Vec<Vec<u16>>> {
        let n_polys = polys.len();
        let n_points = points.len();
        
        if n_polys == 0 || n_points == 0 {
            return Ok(vec![vec![0; n_points]; n_polys]);
        }
        
        // Find the maximum polynomial length
        let max_poly_len = polys.iter().map(|p| p.len()).max().unwrap();
        
        // Flatten polys for C API
        let mut polys_flat = Vec::with_capacity(n_polys * max_poly_len);
        let mut poly_lengths = Vec::with_capacity(n_polys);
        
        for poly in polys {
            poly_lengths.push(poly.len() as i32);
            polys_flat.extend(poly);
            polys_flat.extend(vec![0u16; max_poly_len - poly.len()]);
        }
        
        // Get Galois field tables
        let exp_table = self.galois_field.get_exp_table()?;
        let log_table = self.galois_field.get_log_table()?;
        
        // Create output buffer
        let mut results_flat = vec![0u16; n_polys * n_points];
        
        // Call AVX2 polynomial evaluation
        unsafe {
            avx2_ffi::polynomial_eval_batch(
                polys_flat.as_ptr(),
                poly_lengths.as_ptr(),
                max_poly_len as i32,
                n_polys as i32,
                points.as_ptr(),
                n_points as i32,
                results_flat.as_mut_ptr(),
                log_table.as_ptr(),
                exp_table.as_ptr(),
                self.galois_field.element_count() as i32,
            );
        }
        
        // Reshape results
        let mut results = vec![vec![0u16; n_points]; n_polys];
        for i in 0..n_polys {
            for j in 0..n_points {
                results[i][j] = results_flat[i * n_points + j];
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
            stats.avx2_operations += 1;
        }
        
        Ok(results)
    }
    
    fn perform_tensor_operation(&self, op: TensorOperation) -> Result<()> {
        match op {
            TensorOperation::MatrixMultiply { a, b, c, dims } => {
                self.matrix_multiply_avx2(&a, &b, c, dims)?;
            }
            TensorOperation::ElementWise { input, output, op } => {
                self.element_wise_avx2(&input, output, op)?;
            }
            TensorOperation::Convolution { input, kernel, output, input_dims, kernel_dims, stride, padding } => {
                self.convolution_avx2(
                    &input, &kernel, output,
                    input_dims, kernel_dims, stride, padding
                )?;
            }
        }
        
        Ok(())
    }
    
    fn galois_field(&self) -> Arc<GaloisField> {
        self.galois_field.clone()
    }
    
    fn get_statistics(&self) -> HardwareStatistics {
        self.stats.read().unwrap().clone()
    }
}

/// Helper function to determine available system memory.
/// 
/// Uses platform-specific methods to determine the amount of available memory
/// in the system. Falls back to a conservative estimate based on CPU count
/// if platform-specific methods are not available.
/// 
/// # Returns
/// 
/// Available memory in bytes
fn available_memory() -> usize {
    // Try to get memory info using platform-specific methods
    #[cfg(target_os = "linux")]
    {
        match std::fs::read_to_string("/proc/meminfo") {
            Ok(meminfo) => {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(mem_str) = line.split_whitespace().nth(1) {
                            if let Ok(mem_kb) = mem_str.parse::<usize>() {
                                return mem_kb * 1024;
                            }
                        }
                    }
                }
            }
            Err(_) => {}
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // On Windows, we would use GlobalMemoryStatusEx, but for simplicity and cross-platform
        // compatibility, we'll use a conservative estimate based on the number of CPUs
        let cpus = num_cpus::get();
        return cpus * 1024 * 1024 * 1024; // Estimate 1GB per logical CPU
    }
    
    #[cfg(target_os = "macos")]
    {
        // On macOS, we would use sysctl, but for simplicity and cross-platform
        // compatibility, we'll use a conservative estimate based on the number of CPUs
        let cpus = num_cpus::get();
        return cpus * 1024 * 1024 * 1024; // Estimate 1GB per logical CPU
    }
    
}

/// CPU accelerator using OpenMP for parallel execution.
/// 
/// This implementation leverages OpenMP for multi-threaded parallelism,
/// enabling efficient execution of matrix and vector operations across
/// multiple CPU cores.
#[derive(Debug)]
pub struct OpenMPAccelerator {
    /// Galois field for arithmetic operations
    galois_field: Arc<GaloisField>,
    /// Statistics for hardware operations
    stats: RwLock<HardwareStatistics>,
    /// Number of threads to use (0 means auto)
    num_threads: i32,
}

impl OpenMPAccelerator {
    /// Creates a new OpenMP accelerator.
    ///
    /// # Arguments
    ///
    /// * `galois_field` - Galois field for arithmetic operations
    /// * `num_threads` - Number of threads to use (0 means auto)
    ///
    /// # Returns
    ///
    /// A new OpenMP accelerator
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub fn new(galois_field: Arc<GaloisField>, num_threads: i32) -> Result<Self> {
        Ok(Self {
            galois_field,
            stats: RwLock::new(HardwareStatistics::default()),
            num_threads,
        })
    }
    
    /// Checks if OpenMP accelerator is available on the current system.
    pub fn is_available() -> bool {
        // OpenMP is generally available on all platforms
        // but we'll verify based on the available implementation
        #[cfg(feature = "openmp")]
        {
            true
        }
        
        #[cfg(not(feature = "openmp"))]
        {
            false
        }
    }
    
    /// Matrix multiplication using OpenMP parallelization.
    /// 
    /// Performs optimized matrix multiplication C = A * B using multi-threading.
    ///
    /// # Arguments
    ///
    /// * `a` - First matrix (m x k)
    /// * `b` - Second matrix (k x n)
    /// * `c` - Result matrix (m x n)
    /// * `dims` - Dimensions (m, k, n)
    ///
    /// # Returns
    ///
    /// Result indicating success or an error
    fn matrix_multiply_openmp(
        &self,
        a: &[f32],
        b: &[f32],
        c: Arc<parking_lot::Mutex<Vec<f32>>>,
        dims: (usize, usize, usize),
    ) -> Result<()> {
        let (m, k, n) = dims;
        
        // Create output buffer
        let mut c_data = vec![0.0f32; m * n];
        
        // Call OpenMP matrix multiplication
        unsafe {
            openmp_ffi::openmp_matrix_multiply(
                a.as_ptr(),
                b.as_ptr(),
                c_data.as_mut_ptr(),
                m as i32,
                n as i32,
                k as i32,
                self.num_threads,
            );
        }
        
        // Update output
        let mut c_guard = c.lock();
        *c_guard = c_data;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(())
    }
    
    /// Element-wise operation using OpenMP parallelization.
    /// 
    /// Applies an element-wise operation to a tensor using multi-threading.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `output` - Output tensor
    /// * `op` - Element-wise operation to apply
    ///
    /// # Returns
    ///
    /// Result indicating success or an error
    fn element_wise_openmp(
        &self,
        input: &[f32],
        output: Arc<parking_lot::Mutex<Vec<f32>>>,
        op: ElementWiseOp,
    ) -> Result<()> {
        // Convert ElementWiseOp to OpenMP ElementWiseOperation
        let openmp_op = match op {
            ElementWiseOp::ReLU => openmp_ffi::ElementWiseOperation::RELU,
            ElementWiseOp::Sigmoid => openmp_ffi::ElementWiseOperation::SIGMOID,
            ElementWiseOp::Tanh => openmp_ffi::ElementWiseOperation::TANH,
            ElementWiseOp::Add(_val) => openmp_ffi::ElementWiseOperation::ADD,
            ElementWiseOp::Multiply(_val) => openmp_ffi::ElementWiseOperation::MULTIPLY,
        };
        
        // Get constant value for Add/Multiply operations
        let constant = match op {
            ElementWiseOp::Add(val) => val,
            ElementWiseOp::Multiply(val) => val,
            _ => 0.0,
        };
        
        // Create output buffer
        let mut output_data = vec![0.0f32; input.len()];
        
        // Call OpenMP element-wise operation
        unsafe {
            openmp_ffi::openmp_element_wise_operation(
                input.as_ptr(),
                output_data.as_mut_ptr(),
                input.len() as i32,
                openmp_op,
                constant,
                self.num_threads,
            );
        }
        
        // Update output
        let mut output_guard = output.lock();
        *output_guard = output_data;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(())
    }
    
    /// Convolution using OpenMP parallelization.
    /// 
    /// Performs optimized 2D convolution using multi-threading.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `kernel` - Convolution kernel
    /// * `output` - Output tensor
    /// * `input_dims` - Input dimensions (batch_size, height, width, channels)
    /// * `kernel_dims` - Kernel dimensions (height, width, in_channels, out_channels)
    /// * `stride` - Stride (height, width)
    /// * `padding` - Padding (height, width)
    ///
    /// # Returns
    ///
    /// Result indicating success or an error
    fn convolution_openmp(
        &self,
        input: &[f32],
        kernel: &[f32],
        output: Arc<parking_lot::Mutex<Vec<f32>>>,
        input_dims: (usize, usize, usize, usize),
        kernel_dims: (usize, usize, usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<()> {
        // Extract dimensions
        let (batch_size, input_height, input_width, input_channels) = input_dims;
        let (kernel_height, kernel_width, kernel_in_channels, kernel_out_channels) = kernel_dims;
        let (stride_h, stride_w) = stride;
        let (padding_h, padding_w) = padding;
        
        // Check dimensions
        if input_channels != kernel_in_channels {
            return Err(Error::InvalidInput(format!(
                "Input channels ({}) must match kernel input channels ({})",
                input_channels, kernel_in_channels
            )));
        }
        
        // Calculate output dimensions
        let output_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;
        let output_size = batch_size * output_height * output_width * kernel_out_channels;
        
        // Create output buffer
        let mut output_data = vec![0.0f32; output_size];
        
        // Call OpenMP convolution
        unsafe {
            openmp_ffi::openmp_convolution(
                input.as_ptr(),
                kernel.as_ptr(),
                output_data.as_mut_ptr(),
                batch_size as i32,
                input_height as i32,
                input_width as i32,
                input_channels as i32,
                kernel_height as i32,
                kernel_width as i32,
                kernel_out_channels as i32,
                output_height as i32,
                output_width as i32,
                stride_h as i32,
                stride_w as i32,
                padding_h as i32,
                padding_w as i32,
                self.num_threads,
            );
        }
        
        // Update output
        let mut output_guard = output.lock();
        *output_guard = output_data;
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(())
    }
}

impl HardwareAccelerator for OpenMPAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::Cpu
    }
    
    fn is_available(&self) -> bool {
        // OpenMP is generally available on all platforms
        // but we'll verify based on the available implementation
        #[cfg(feature = "openmp")]
        {
            true
        }
        
        #[cfg(not(feature = "openmp"))]
        {
            false
        }
    }
    
    fn capabilities(&self) -> HardwareCapabilities {
        HardwareCapabilities {
            avx2_available: false,
            openmp_available: true,
            cuda_available: false,
            cuda_device_count: 0,
            opencl_available: false,
            opencl_device_count: 0,
            processor_count: num_cpus::get(),
            available_memory: available_memory(),
            hamming_available: true,
            bch_available: true,
            reed_muller_available: true,
            turbo_available: false,
            convolutional_available: true,
            fountain_available: false,
        }
    }
    
    fn generate_lookup_tables(&self, path: &Path) -> Result<()> {
        // Create the OpenMP-specific tables directory
        let openmp_path = path.join("openmp");
        std::fs::create_dir_all(&openmp_path)?;
        
        // Generate Galois field lookup tables
        self.galois_field().generate_lookup_tables()?;
        
        Ok(())
    }
    
    fn calculate_syndromes(&self, data: &[u8], syndrome_count: usize) -> Result<Vec<u16>> {
        // Convert data to u16
        let data_u16: Vec<u16> = data.iter().map(|&x| x as u16).collect();
        
        // Get Galois field tables
        let exp_table = self.galois_field.get_exp_table()?;
        let log_table = self.galois_field.get_log_table()?;
        
        // Create output buffer
        let mut syndromes = vec![0u16; syndrome_count];
        
        // Call OpenMP syndrome calculation
        unsafe {
            openmp_ffi::openmp_calculate_syndromes(
                data_u16.as_ptr(),
                data_u16.len() as i32,
                syndromes.as_mut_ptr(),
                syndrome_count as i32,
                exp_table.as_ptr(),
                log_table.as_ptr(),
                self.galois_field.element_count() as i32,
                self.num_threads,
            );
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(syndromes)
    }
    
    fn multiply_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput("Input arrays must have the same length".into()));
        }
        
        // Get Galois field tables
        let exp_table = self.galois_field.get_exp_table()?;
        let log_table = self.galois_field.get_log_table()?;
        
        // Create output buffer
        let mut result = vec![0u16; a.len()];
        
        // Call OpenMP Galois field multiplication
        unsafe {
            openmp_ffi::openmp_gf_multiply(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                a.len() as i32,
                exp_table.as_ptr(),
                log_table.as_ptr(),
                self.galois_field.element_count() as i32,
                self.num_threads,
            );
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(result)
    }
    
    fn add_vec(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(Error::InvalidInput("Input arrays must have the same length".into()));
        }
        
        // Create output buffer
        let mut result = vec![0u16; a.len()];
        
        // Call OpenMP Galois field addition
        unsafe {
            openmp_ffi::openmp_gf_add(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                a.len() as i32,
                self.num_threads,
            );
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(result)
    }
    
    fn polynomial_eval_batch(&self, polys: &[Vec<u16>], points: &[u16]) -> Result<Vec<Vec<u16>>> {
        let n_polys = polys.len();
        let n_points = points.len();
        
        if n_polys == 0 || n_points == 0 {
            return Ok(vec![vec![0; n_points]; n_polys]);
        }
        
        // Find the maximum polynomial length
        let max_poly_len = polys.iter().map(|p| p.len()).max().unwrap();
        
        // Flatten polys for C API
        let mut polys_flat = Vec::with_capacity(n_polys * max_poly_len);
        let mut poly_lengths = Vec::with_capacity(n_polys);
        
        for poly in polys {
            poly_lengths.push(poly.len() as i32);
            polys_flat.extend(poly);
            polys_flat.extend(vec![0u16; max_poly_len - poly.len()]);
        }
        
        // Get Galois field tables
        let exp_table = self.galois_field.get_exp_table()?;
        let log_table = self.galois_field.get_log_table()?;
        
        // Create output buffer
        let mut results_flat = vec![0u16; n_polys * n_points];
        
        // Call OpenMP polynomial evaluation
        unsafe {
            openmp_ffi::openmp_polynomial_eval_batch(
                polys_flat.as_ptr(),
                poly_lengths.as_ptr(),
                max_poly_len as i32,
                n_polys as i32,
                points.as_ptr(),
                n_points as i32,
                results_flat.as_mut_ptr(),
                log_table.as_ptr(),
                exp_table.as_ptr(),
                self.galois_field.element_count() as i32,
                self.num_threads,
            );
        }
        
        // Reshape results
        let mut results = vec![vec![0u16; n_points]; n_polys];
        for i in 0..n_polys {
            for j in 0..n_points {
                results[i][j] = results_flat[i * n_points + j];
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.cpu_operations += 1;
        }
        
        Ok(results)
    }
    
    fn perform_tensor_operation(&self, op: TensorOperation) -> Result<()> {
        match op {
            TensorOperation::MatrixMultiply { a, b, c, dims } => {
                self.matrix_multiply_openmp(&a, &b, c, dims)?;
            }
            TensorOperation::ElementWise { input, output, op } => {
                self.element_wise_openmp(&input, output, op)?;
            }
            TensorOperation::Convolution { input, kernel, output, input_dims, kernel_dims, stride, padding } => {
                self.convolution_openmp(
                    &input, &kernel, output,
                    input_dims, kernel_dims, stride, padding
                )?;
            }
        }
        
        Ok(())
    }
    
    fn galois_field(&self) -> Arc<GaloisField> {
        self.galois_field.clone()
    }
    
    fn get_statistics(&self) -> HardwareStatistics {
        self.stats.read().unwrap().clone()
    }
}

/// Register CPU hardware accelerators.
/// 
/// This function initializes and registers all available CPU-based hardware 
/// accelerators (AVX2 and OpenMP) to the provided accelerator list.
///
/// # Arguments
///
/// * `accelerators` - List to add the CPU accelerators to
///
/// # Returns
///
/// Result indicating success or an error
pub fn register_cpu_accelerators(accelerators: &mut Vec<Arc<dyn HardwareAccelerator>>) -> Result<()> {
    // Check if CPU acceleration is available
    if CPUAccelerator::is_available() {
        // Create Galois field
        let galois_field = Arc::new(GaloisField::new(0x11D));
        
        // Create CPU accelerator
        let cpu_accelerator = Arc::new(CPUAccelerator::new(galois_field.clone())?);
        
        // Register CPU accelerator
        accelerators.push(cpu_accelerator.clone());
        
        // Check if OpenMP is available
        if OpenMPAccelerator::is_available() {
            // Create OpenMP accelerator (auto thread count)
            let openmp_accelerator = Arc::new(OpenMPAccelerator::new(galois_field, 0)?);
            
            // Register OpenMP accelerator
            accelerators.push(openmp_accelerator);
        }
    }
    
    Ok(())
}