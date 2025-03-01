//! Galois Field implementation for finite field arithmetic.
//!
//! This module provides a complete implementation of Galois Fields (GF(2^m))
//! for finite field arithmetic operations required by error correction codes.
//! It includes optimized implementations for common operations like addition,
//! multiplication, division, and polynomial evaluation.

use std::path::Path;
use std::sync::RwLock;

use lazy_static::lazy_static;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Error, Result};

/// The maximum supported field size.
pub const MAX_FIELD_SIZE: usize = 1 << 16; // GF(2^16)

/// A Galois Field implementation for finite field arithmetic.
#[derive(Debug)]
pub struct GaloisField {
    /// The irreducible polynomial that defines the field.
    field_polynomial: u32,
    
    /// The size of the field.
    field_size: usize,
    
    /// The number of elements in the multiplicative group of the field.
    element_count: usize,
    
    /// Exponential table for fast multiplication (maps power of primitive element to field element).
    exp_table: RwLock<Vec<u16>>,
    
    /// Logarithm table for fast division (maps field element to power of primitive element).
    log_table: RwLock<Vec<u16>>,
    
    /// Cache for multiplication results to avoid recomputation.
    mult_cache: Mutex<HashMap<(u16, u16), u16>>,
    
    /// Whether to use AVX2 acceleration when available.
    use_avx2: bool,
    
    /// Whether tables have been normalized.
    tables_normalized: RwLock<bool>,
}

/// Precomputed tables for a Galois Field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaloisTables {
    /// Field polynomial
    field_polynomial: u32,
    
    /// Field size
    field_size: usize,
    
    /// Number of elements in the multiplicative group
    element_count: usize,
    
    /// Exponential table
    exp_table: Vec<u16>,
    
    /// Logarithm table
    log_table: Vec<u16>,
}

// Static cache for field tables to avoid recomputation
lazy_static! {
    static ref FIELD_TABLES_CACHE: Mutex<HashMap<u32, GaloisTables>> = Mutex::new(HashMap::new());
}

impl GaloisField {
    /// Creates a new Galois Field with the specified irreducible polynomial.
    ///
    /// # Arguments
    ///
    /// * `field_polynomial` - An irreducible polynomial that defines the field,
    ///                       represented as an integer (e.g., 0x11D for GF(2^8))
    ///
    /// # Returns
    ///
    /// A new Galois Field instance
    ///
    /// # Errors
    ///
    /// Returns an error if the field polynomial is invalid
    pub fn new(field_polynomial: u32) -> Self {
        // Validate input polynomial
        if field_polynomial <= 0 {
            panic!("Field polynomial must be positive");
        }
        
        // Determine field size from the polynomial degree
        let degree = 32 - field_polynomial.leading_zeros();
        let field_size = 1 << (degree - 1);
        let element_count = field_size - 1;
        
        // Check if tables are in the cache
        let cache = FIELD_TABLES_CACHE.lock();
        if let Some(tables) = cache.get(&field_polynomial) {
            // Use cached tables
            return Self {
                field_polynomial,
                field_size,
                element_count,
                exp_table: RwLock::new(tables.exp_table.clone()),
                log_table: RwLock::new(tables.log_table.clone()),
                mult_cache: Mutex::new(HashMap::with_capacity(10000)),
                use_avx2: is_avx2_supported(),
                tables_normalized: RwLock::new(true),
            };
        }
        drop(cache); // Release the lock before generating tables
        
        // Initialize tables
        let exp_table = vec![0; 2 * element_count];
        let log_table = vec![0; field_size];
        
        let field = Self {
            field_polynomial,
            field_size,
            element_count,
            exp_table: RwLock::new(exp_table),
            log_table: RwLock::new(log_table),
            mult_cache: Mutex::new(HashMap::with_capacity(10000)),
            use_avx2: is_avx2_supported(),
            tables_normalized: RwLock::new(false),
        };
        
        // Generate tables
        field.generate_tables();
        
        // Cache the tables
        let tables = GaloisTables {
            field_polynomial,
            field_size,
            element_count,
            exp_table: field.exp_table.read().expect("Failed to read exp_table").clone(),
            log_table: field.log_table.read().expect("Failed to read log_table").clone(),
        };
        FIELD_TABLES_CACHE.lock().insert(field_polynomial, tables);
        
        field
    }
    
    /// Generates the exponential and logarithm lookup tables for field arithmetic.
    ///
    /// This precomputes values to make multiplication and division operations
    /// much more efficient during encoding and decoding operations.
    fn generate_tables(&self) {
        let mut exp_table = self.exp_table.write().expect("Failed to write to exp_table");
        let mut log_table = self.log_table.write().expect("Failed to write to log_table");
        
        let mut x = 1;
        for i in 0..self.element_count {
            exp_table[i] = x as u16;
            
            // Calculate the next value by multiplying by the primitive element (x)
            x <<= 1;
            if x & self.field_size != 0 {
                x ^= self.field_polynomial as usize;
            }
        }
        
        // Extend the exp table to handle wrap-around
        for i in self.element_count..(2 * self.element_count) {
            exp_table[i] = exp_table[i - self.element_count];
        }
        
        // Generate log table
        log_table[0] = 0; // log(0) is undefined, but we set it to 0 for convenience
        for i in 0..self.element_count {
            log_table[exp_table[i] as usize] = i as u16;
        }
        
        // Mark tables as normalized
        *self.tables_normalized.write().expect("Failed to write to tables_normalized") = true;
    }
    
    /// Normalizes the exp and log tables to ensure consistency and handle edge cases.
    ///
    /// This should be called after generating tables if additional validation is needed.
    fn normalize_tables(&self) {
        if *self.tables_normalized.read().expect("Failed to read tables_normalized") {
            return;
        }
        
        let mut exp_table = self.exp_table.write().expect("Failed to write to exp_table");
        let mut log_table = self.log_table.write().expect("Failed to write to log_table");
        
        // Ensure log of 0 is handled consistently
        log_table[0] = 0;
        
        // Ensure the exp table is properly extended
        for i in self.element_count..(2 * self.element_count) {
            exp_table[i] = exp_table[i - self.element_count];
        }
        
        // Validate that exp and log are inverses
        for i in 1..self.element_count {
            let value = exp_table[i];
            if log_table[value as usize] != i as u16 {
                tracing::warn!(
                    "Inconsistency in Galois Field tables: exp({})={}, log({})={}",
                    i, value, value, log_table[value as usize]
                );
                log_table[value as usize] = i as u16;
            }
        }
        
        *self.tables_normalized.write().expect("Failed to write to tables_normalized") = true;
    }
    
    /// Generates lookup tables and saves them to disk for faster initialization.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save lookup tables
    ///
    /// # Returns
    ///
    /// Ok(()) if successful, or an error if table generation fails
    pub fn generate_lookup_tables(&self) -> Result<()> {
        // Ensure tables are generated and normalized
        self.normalize_tables();
        
        // Create tables directory if it doesn't exist
        std::fs::create_dir_all(Path::new("tables"))?;
        
        // Create tables structure
        let tables = GaloisTables {
            field_polynomial: self.field_polynomial,
            field_size: self.field_size,
            element_count: self.element_count,
            exp_table: self.exp_table.read().expect("Failed to read exp_table").clone(),
            log_table: self.log_table.read().expect("Failed to read log_table").clone(),
        };
        
        // Save tables to file
        let file_name = format!("tables/gf_{:x}.bin", self.field_polynomial);
        let file = std::fs::File::create(file_name)?;
        bincode::serialize_into(file, &tables)
            .map_err(|e| Error::BinarySerialization(e))?;
        
        Ok(())
    }
    
    /// Loads lookup tables from disk if available.
    ///
    /// # Arguments
    ///
    /// * `field_polynomial` - Field polynomial for the Galois Field
    ///
    /// # Returns
    ///
    /// Galois tables if loaded successfully, or None if not found
    pub fn load_lookup_tables(field_polynomial: u32) -> Option<GaloisTables> {
        let file_name = format!("tables/gf_{:x}.bin", field_polynomial);
        match std::fs::File::open(file_name) {
            Ok(file) => {
                match bincode::deserialize_from(file) {
                    Ok(tables) => Some(tables),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    }
    
    /// Returns the field polynomial.
    pub fn field_polynomial(&self) -> u32 {
        self.field_polynomial
    }
    
    /// Returns the field size.
    pub fn field_size(&self) -> usize {
        self.field_size
    }
    
    /// Returns the number of elements in the multiplicative group.
    pub fn element_count(&self) -> usize {
        self.element_count
    }
    
    /// Returns a copy of the exponential table for hardware acceleration.
    pub fn get_exp_table(&self) -> Result<Vec<u16>> {
        let exp_table = self.exp_table.read().expect("Failed to read exp_table").clone();
        Ok(exp_table)
    }
    
    /// Returns a copy of the logarithm table for hardware acceleration.
    pub fn get_log_table(&self) -> Result<Vec<u16>> {
        let log_table = self.log_table.read().expect("Failed to read log_table").clone();
        Ok(log_table)
    }
    
    /// Add two elements in the Galois field (which is XOR in binary fields).
    ///
    /// # Arguments
    ///
    /// * `a` - First field element
    /// * `b` - Second field element
    ///
    /// # Returns
    ///
    /// The sum in the Galois field
    #[inline]
    pub fn add(&self, a: u16, b: u16) -> u16 {
        a ^ b
    }
    
    /// Subtract two elements in the Galois field (same as addition in binary fields).
    ///
    /// # Arguments
    ///
    /// * `a` - First field element
    /// * `b` - Second field element
    ///
    /// # Returns
    ///
    /// The difference in the Galois field
    #[inline]
    pub fn subtract(&self, a: u16, b: u16) -> u16 {
        self.add(a, b)
    }
    
    /// Multiply two elements in the Galois field using lookup tables.
    ///
    /// # Arguments
    ///
    /// * `a` - First field element
    /// * `b` - Second field element
    ///
    /// # Returns
    ///
    /// The product in the Galois field
    #[inline]
    pub fn multiply(&self, a: u16, b: u16) -> u16 {
        // Handle special cases for 0
        if a == 0 || b == 0 {
            return 0;
        }
        
        // Check cache for common multiplications
        let cache_key = if a <= b { (a, b) } else { (b, a) }; // Commutative operation
        {
            let cache = self.mult_cache.lock();
            if let Some(&result) = cache.get(&cache_key) {
                return result;
            }
        }
        
        // Use table lookup for multiplication
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        let log_table = self.log_table.read().expect("Failed to read log_table");
        
        let log_a = log_table[a as usize];
        let log_b = log_table[b as usize];
        let log_sum = (log_a as usize + log_b as usize) % self.element_count;
        let result = exp_table[log_sum];
        
        // Cache result for future use
        if self.mult_cache.lock().len() < 10000 {
            self.mult_cache.lock().insert(cache_key, result);
        }
        
        result
    }
    
    /// Multiply two elements using direct polynomial multiplication without tables.
    ///
    /// This is slower but can be useful for verification or when tables are not available.
    ///
    /// # Arguments
    ///
    /// * `a` - First field element
    /// * `b` - Second field element
    ///
    /// # Returns
    ///
    /// The product in the Galois field
    pub fn multiply_direct(&self, a: u16, b: u16) -> u16 {
        if a == 0 || b == 0 {
            return 0;
        }
        
        let mut result = 0u16;
        let mut a_temp = a as u32;
        let mut b_temp = b as u32;
        
        while b_temp > 0 {
            if b_temp & 1 != 0 {
                result ^= a_temp as u16;
            }
            
            a_temp <<= 1;
            if a_temp & (self.field_size as u32) != 0 {
                a_temp ^= self.field_polynomial;
            }
            
            b_temp >>= 1;
        }
        
        result
    }
    
    /// Multiply arrays of field elements element-wise.
    ///
    /// # Arguments
    ///
    /// * `a` - First array of field elements
    /// * `b` - Second array of field elements
    ///
    /// # Returns
    ///
    /// Array containing element-wise products
    pub fn multiply_vec(&self, a: &[u16], b: &[u16]) -> Vec<u16> {
        assert_eq!(a.len(), b.len(), "Input arrays must have the same length");
        
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        let log_table = self.log_table.read().expect("Failed to read log_table");
        
        // Use AVX2 if available and enabled
        #[cfg(feature = "cpu")]
        if self.use_avx2 && is_avx2_supported() && a.len() >= 8 {
            return self.multiply_vec_avx2(a, b);
        }
        
        // Standard vectorized implementation
        let mut result = vec![0u16; a.len()];
        for i in 0..a.len() {
            if a[i] == 0 || b[i] == 0 {
                result[i] = 0;
            } else {
                let log_a = log_table[a[i] as usize];
                let log_b = log_table[b[i] as usize];
                let log_sum = (log_a as usize + log_b as usize) % self.element_count;
                result[i] = exp_table[log_sum];
            }
        }
        
        result
    }
    
    /// Multiply arrays of field elements element-wise using AVX2 instructions.
    #[cfg(feature = "cpu")]
    fn multiply_vec_avx2(&self, a: &[u16], b: &[u16]) -> Vec<u16> {
        use std::arch::x86_64::*;
        
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        let log_table = self.log_table.read().expect("Failed to read log_table");
        let element_count = self.element_count;
        
        let mut result = vec![0u16; a.len()];
        
        // Safety: We've already checked that AVX2 is available
        unsafe {
            // Process 16 elements at a time (256-bit AVX2 registers)
            let chunk_size = 16;
            let full_chunks = a.len() / chunk_size;
            
            for i in 0..full_chunks {
                let offset = i * chunk_size;
                
                // Load 16 elements from a and b
                let a_chunk = _mm256_loadu_si256(a[offset..].as_ptr() as *const __m256i);
                let b_chunk = _mm256_loadu_si256(b[offset..].as_ptr() as *const __m256i);
                
                // Create zero mask
                let zero = _mm256_setzero_si256();
                let a_is_zero = _mm256_cmpeq_epi16(a_chunk, zero);
                let b_is_zero = _mm256_cmpeq_epi16(b_chunk, zero);
                let _either_zero = _mm256_or_si256(a_is_zero, b_is_zero);
                
                // Extract values from SIMD registers into arrays for processing
                let mut a_vals = [0u16; 16];
                let mut b_vals = [0u16; 16];
                let mut prod_vals = [0i16; 16];
                
                // Store to arrays first (avoiding direct dynamic indices in SIMD intrinsics)
                for j in 0..chunk_size {
                    // For j=0, use const 0, for j=1 use const 1, etc.
                    a_vals[j] = match j {
                        0 => _mm256_extract_epi16(a_chunk, 0) as u16,
                        1 => _mm256_extract_epi16(a_chunk, 1) as u16,
                        2 => _mm256_extract_epi16(a_chunk, 2) as u16,
                        3 => _mm256_extract_epi16(a_chunk, 3) as u16,
                        4 => _mm256_extract_epi16(a_chunk, 4) as u16,
                        5 => _mm256_extract_epi16(a_chunk, 5) as u16,
                        6 => _mm256_extract_epi16(a_chunk, 6) as u16,
                        7 => _mm256_extract_epi16(a_chunk, 7) as u16,
                        8 => _mm256_extract_epi16(a_chunk, 8) as u16,
                        9 => _mm256_extract_epi16(a_chunk, 9) as u16,
                        10 => _mm256_extract_epi16(a_chunk, 10) as u16,
                        11 => _mm256_extract_epi16(a_chunk, 11) as u16,
                        12 => _mm256_extract_epi16(a_chunk, 12) as u16,
                        13 => _mm256_extract_epi16(a_chunk, 13) as u16,
                        14 => _mm256_extract_epi16(a_chunk, 14) as u16,
                        15 => _mm256_extract_epi16(a_chunk, 15) as u16,
                        _ => unreachable!(),
                    };
                    
                    b_vals[j] = match j {
                        0 => _mm256_extract_epi16(b_chunk, 0) as u16,
                        1 => _mm256_extract_epi16(b_chunk, 1) as u16,
                        2 => _mm256_extract_epi16(b_chunk, 2) as u16,
                        3 => _mm256_extract_epi16(b_chunk, 3) as u16,
                        4 => _mm256_extract_epi16(b_chunk, 4) as u16,
                        5 => _mm256_extract_epi16(b_chunk, 5) as u16,
                        6 => _mm256_extract_epi16(b_chunk, 6) as u16,
                        7 => _mm256_extract_epi16(b_chunk, 7) as u16,
                        8 => _mm256_extract_epi16(b_chunk, 8) as u16,
                        9 => _mm256_extract_epi16(b_chunk, 9) as u16,
                        10 => _mm256_extract_epi16(b_chunk, 10) as u16,
                        11 => _mm256_extract_epi16(b_chunk, 11) as u16,
                        12 => _mm256_extract_epi16(b_chunk, 12) as u16,
                        13 => _mm256_extract_epi16(b_chunk, 13) as u16,
                        14 => _mm256_extract_epi16(b_chunk, 14) as u16,
                        15 => _mm256_extract_epi16(b_chunk, 15) as u16,
                        _ => unreachable!(),
                    };
                    
                    // Calculate product
                    let prod = if a_vals[j] == 0 || b_vals[j] == 0 {
                        0
                    } else {
                        let log_a = log_table[a_vals[j] as usize];
                        let log_b = log_table[b_vals[j] as usize];
                        let log_sum = (log_a as usize + log_b as usize) % element_count;
                        exp_table[log_sum]
                    };
                    
                    prod_vals[j] = prod as i16; // Proper cast to i16
                }
                
                // Now rebuild the result vector using constant indices with proper casting
                let mut result_chunk = _mm256_setzero_si256(); // Start with zero
                
                // Convert i16 to i32 for the first parameter, use fixed i32 constants for the second parameter
                // The compiler expects an i16 for the value but the function actually takes an i32
                // Use .try_into().unwrap() as suggested by the compiler to ensure safe conversion
                let val0: i32 = prod_vals[0].into();
                let val1: i32 = prod_vals[1].into();
                let val2: i32 = prod_vals[2].into();
                let val3: i32 = prod_vals[3].into();
                let val4: i32 = prod_vals[4].into();
                let val5: i32 = prod_vals[5].into();
                let val6: i32 = prod_vals[6].into();
                let val7: i32 = prod_vals[7].into();
                let val8: i32 = prod_vals[8].into();
                let val9: i32 = prod_vals[9].into();
                let val10: i32 = prod_vals[10].into();
                let val11: i32 = prod_vals[11].into();
                let val12: i32 = prod_vals[12].into();
                let val13: i32 = prod_vals[13].into();
                let val14: i32 = prod_vals[14].into();
                let val15: i32 = prod_vals[15].into();
                
                result_chunk = _mm256_insert_epi16(result_chunk, val0.try_into().unwrap(), 0);
                result_chunk = _mm256_insert_epi16(result_chunk, val1.try_into().unwrap(), 1);
                result_chunk = _mm256_insert_epi16(result_chunk, val2.try_into().unwrap(), 2);
                result_chunk = _mm256_insert_epi16(result_chunk, val3.try_into().unwrap(), 3);
                result_chunk = _mm256_insert_epi16(result_chunk, val4.try_into().unwrap(), 4);
                result_chunk = _mm256_insert_epi16(result_chunk, val5.try_into().unwrap(), 5);
                result_chunk = _mm256_insert_epi16(result_chunk, val6.try_into().unwrap(), 6);
                result_chunk = _mm256_insert_epi16(result_chunk, val7.try_into().unwrap(), 7);
                result_chunk = _mm256_insert_epi16(result_chunk, val8.try_into().unwrap(), 8);
                result_chunk = _mm256_insert_epi16(result_chunk, val9.try_into().unwrap(), 9);
                result_chunk = _mm256_insert_epi16(result_chunk, val10.try_into().unwrap(), 10);
                result_chunk = _mm256_insert_epi16(result_chunk, val11.try_into().unwrap(), 11);
                result_chunk = _mm256_insert_epi16(result_chunk, val12.try_into().unwrap(), 12);
                result_chunk = _mm256_insert_epi16(result_chunk, val13.try_into().unwrap(), 13);
                result_chunk = _mm256_insert_epi16(result_chunk, val14.try_into().unwrap(), 14);
                result_chunk = _mm256_insert_epi16(result_chunk, val15.try_into().unwrap(), 15);
                
                // Store result
                _mm256_storeu_si256(result[offset..].as_mut_ptr() as *mut __m256i, result_chunk);
            }
            
            // Process remaining elements
            let remaining_start = full_chunks * chunk_size;
            for i in remaining_start..a.len() {
                result[i] = if a[i] == 0 || b[i] == 0 {
                    0
                } else {
                    let log_a = log_table[a[i] as usize];
                    let log_b = log_table[b[i] as usize];
                    let log_sum = (log_a as usize + log_b as usize) % element_count;
                    exp_table[log_sum]
                };
            }
        }
        
        result
    }
    
    /// Divide one element by another in the Galois field.
    ///
    /// # Arguments
    ///
    /// * `a` - Numerator
    /// * `b` - Denominator (must be non-zero)
    ///
    /// # Returns
    ///
    /// The quotient in the Galois field
    ///
    /// # Panics
    ///
    /// Panics if the denominator is zero
    pub fn divide(&self, a: u16, b: u16) -> u16 {
        assert!(b != 0, "Division by zero in Galois Field");
        
        if a == 0 {
            return 0;
        }
        
        let log_table = self.log_table.read().expect("Failed to read log_table");
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        
        let log_a = log_table[a as usize];
        let log_b = log_table[b as usize];
        let mut log_diff = log_a as i32 - log_b as i32;
        
        // Ensure the result is in the valid range
        if log_diff < 0 {
            log_diff += self.element_count as i32;
        }
        
        exp_table[log_diff as usize]
    }
    
    /// Raise a field element to a power.
    ///
    /// # Arguments
    ///
    /// * `a` - Base element
    /// * `n` - Exponent
    ///
    /// # Returns
    ///
    /// a^n in the Galois field
    pub fn power(&self, a: u16, n: u32) -> u16 {
        if a == 0 {
            return if n > 0 { 0 } else { 1 }; // 0^0 = 1 by convention
        }
        
        let log_table = self.log_table.read().expect("Failed to read log_table");
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        
        let log_a = log_table[a as usize];
        let log_result = ((log_a as u64 * n as u64) % self.element_count as u64) as usize;
        
        exp_table[log_result]
    }
    
    /// Compute the exponential of the primitive element to the given power.
    ///
    /// # Arguments
    ///
    /// * `power` - The exponent
    ///
    /// # Returns
    ///
    /// The exponential result
    pub fn exp(&self, power: i32) -> u16 {
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        
        let power = if power < 0 {
            (power % self.element_count as i32) + self.element_count as i32
        } else {
            power % self.element_count as i32
        };
        
        exp_table[power as usize]
    }
    
    /// Compute the logarithm of a field element.
    ///
    /// # Arguments
    ///
    /// * `value` - A non-zero field element
    ///
    /// # Returns
    ///
    /// The logarithm of the value
    ///
    /// # Panics
    ///
    /// Panics if the value is zero (logarithm is undefined)
    pub fn log(&self, value: u16) -> u16 {
        assert!(value != 0, "Cannot compute logarithm of zero in Galois Field");
        assert!(value < self.field_size as u16, "Value is outside the field range");
        
        let log_table = self.log_table.read().expect("Failed to read log_table");
        log_table[value as usize]
    }
    
    /// Compute the multiplicative inverse of a field element.
    ///
    /// # Arguments
    ///
    /// * `value` - A non-zero field element
    ///
    /// # Returns
    ///
    /// The multiplicative inverse
    ///
    /// # Panics
    ///
    /// Panics if the value is zero (inverse is undefined)
    pub fn inverse(&self, value: u16) -> u16 {
        assert!(value != 0, "Zero has no multiplicative inverse in Galois Field");
        
        let log_table = self.log_table.read().expect("Failed to read log_table");
        let exp_table = self.exp_table.read().expect("Failed to read exp_table");
        
        let log_value = log_table[value as usize];
        let inverse_power = self.element_count - log_value as usize;
        
        exp_table[inverse_power]
    }
    
    /// Evaluate a polynomial at a specific point using Horner's method.
    ///
    /// # Arguments
    ///
    /// * `poly` - Polynomial coefficients (highest degree first)
    /// * `x` - Point at which to evaluate
    ///
    /// # Returns
    ///
    /// The polynomial evaluation result
    pub fn polynomial_eval(&self, poly: &[u16], x: u16) -> u16 {
        if poly.is_empty() {
            return 0;
        }
        
        // Use Horner's method for efficient evaluation
        let mut result = poly[0];
        
        for &coef in poly.iter().skip(1) {
            result = self.add(self.multiply(result, x), coef);
        }
        
        result
    }
    
    /// Evaluate multiple polynomials at multiple points.
    ///
    /// # Arguments
    ///
    /// * `polys` - Vector of polynomial coefficients (highest degree first)
    /// * `points` - Vector of points at which to evaluate
    ///
    /// # Returns
    ///
    /// 2D vector of results [poly_idx][point_idx]
    pub fn polynomial_eval_batch(&self, polys: &[Vec<u16>], points: &[u16]) -> Vec<Vec<u16>> {
        let n_polys = polys.len();
        let n_points = points.len();
        
        let mut results = vec![vec![0u16; n_points]; n_polys];
        
        // Use parallel computation if available
        #[cfg(feature = "cpu")]
        {
            use rayon::prelude::*;
            
            results.par_iter_mut().enumerate().for_each(|(i, result_row)| {
                for (j, point) in points.iter().enumerate() {
                    result_row[j] = self.polynomial_eval(&polys[i], *point);
                }
            });
        }
        
        // Sequential evaluation if parallel computation is not available
        #[cfg(not(feature = "cpu"))]
        {
            for i in 0..n_polys {
                for j in 0..n_points {
                    results[i][j] = self.polynomial_eval(&polys[i], points[j]);
                }
            }
        }
        
        results
    }
    
    /// Add two polynomials in the Galois field.
    ///
    /// # Arguments
    ///
    /// * `poly1` - First polynomial coefficients (highest degree first)
    /// * `poly2` - Second polynomial coefficients (highest degree first)
    ///
    /// # Returns
    ///
    /// The sum polynomial
    pub fn polynomial_add(&self, poly1: &[u16], poly2: &[u16]) -> Vec<u16> {
        // Ensure poly1 is longer or equal to poly2
        if poly1.len() < poly2.len() {
            return self.polynomial_add(poly2, poly1);
        }
        
        let mut result = poly1.to_vec();
        
        // Add coefficients
        for (i, &coef2) in poly2.iter().enumerate() {
            let result_idx = result.len() - poly2.len() + i;
            result[result_idx] = self.add(result[result_idx], coef2);
        }
        
        result
    }
    
    /// Multiply two polynomials in the Galois field.
    ///
    /// # Arguments
    ///
    /// * `poly1` - First polynomial coefficients (highest degree first)
    /// * `poly2` - Second polynomial coefficients (highest degree first)
    ///
    /// # Returns
    ///
    /// The product polynomial
    pub fn polynomial_multiply(&self, poly1: &[u16], poly2: &[u16]) -> Vec<u16> {
        if poly1.is_empty() || poly2.is_empty() {
            return vec![];
        }
        
        let result_len = poly1.len() + poly2.len() - 1;
        let mut result = vec![0u16; result_len];
        
        for (i, &coef1) in poly1.iter().enumerate() {
            for (j, &coef2) in poly2.iter().enumerate() {
                let idx = i + j;
                let term = self.multiply(coef1, coef2);
                result[idx] = self.add(result[idx], term);
            }
        }
        
        result
    }
    
    /// Clear the multiplication cache to free memory.
    pub fn clear_cache(&self) {
        self.mult_cache.lock().clear();
    }
    
    /// Calculate syndromes for error detection in the received message.
    ///
    /// # Arguments
    ///
    /// * `received` - Received message with potential errors
    /// * `syndrome_count` - Number of syndromes to calculate
    ///
    /// # Returns
    ///
    /// Vector of syndrome values (all zeros means no errors)
    pub fn calculate_syndromes(&self, received: &[u8], syndrome_count: usize) -> Result<Vec<u16>> {
        if received.len() <= syndrome_count {
            return Err(Error::InvalidInput(
                "Received data is too short for syndrome calculation".to_string(),
            ));
        }
        
        let received_u16: Vec<u16> = received.iter().map(|&b| b as u16).collect();
        let mut syndromes = vec![0u16; syndrome_count];
        
        // Calculate each syndrome by evaluating the received polynomial at α^i
        for i in 0..syndrome_count {
            // Horner's method for polynomial evaluation
            let mut result = 0u16;
            let x_i = self.exp(i as i32);
            
            for &r in &received_u16 {
                result = self.add(self.multiply(result, x_i), r);
            }
            
            syndromes[i] = result;
        }
        
        Ok(syndromes)
    }
}

/// Check if AVX2 is supported on the current CPU.
#[cfg(feature = "cpu")]
fn is_avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe {
            // Check if AVX2 is supported
            let cpuid = __cpuid(7);
            (cpuid.ebx & (1 << 5)) != 0
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(not(feature = "cpu"))]
fn is_avx2_supported() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_galois_field_creation() {
        let gf = GaloisField::new(0x11D); // GF(2^8)
        assert_eq!(gf.field_size(), 256);
        assert_eq!(gf.element_count(), 255);
    }
    
    #[test]
    fn test_addition() {
        let gf = GaloisField::new(0x11D);
        assert_eq!(gf.add(0x53, 0xCA), 0x99); // 83 + 202 = 153 (XOR)
        assert_eq!(gf.add(0, 0x5A), 0x5A);
        assert_eq!(gf.add(0x5A, 0), 0x5A);
        assert_eq!(gf.add(0, 0), 0);
    }
    
    #[test]
    fn test_multiplication() {
        let gf = GaloisField::new(0x11D);
        
        // Test some known values
        assert_eq!(gf.multiply(3, 7), 9);
        assert_eq!(gf.multiply(0, 5), 0);
        assert_eq!(gf.multiply(5, 0), 0);
        
        // Test commutative property
        for a in 0..10 {
            for b in 0..10 {
                assert_eq!(gf.multiply(a, b), gf.multiply(b, a));
            }
        }
        
        // Test with direct multiplication
        for a in 0..10 {
            for b in 0..10 {
                assert_eq!(gf.multiply(a, b), gf.multiply_direct(a, b));
            }
        }
    }
    
    #[test]
    fn test_vector_multiplication() {
        let gf = GaloisField::new(0x11D);
        
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![6, 7, 8, 9, 10];
        let expected: Vec<u16> = a.iter().zip(b.iter())
            .map(|(&x, &y)| gf.multiply(x, y))
            .collect();
        
        assert_eq!(gf.multiply_vec(&a, &b), expected);
    }
    
    #[test]
    fn test_division() {
        let gf = GaloisField::new(0x11D);
        
        // Test some known values
        for a in 1..10 {
            for b in 1..10 {
                let c = gf.multiply(a, b);
                assert_eq!(gf.divide(c, b), a);
                assert_eq!(gf.divide(c, a), b);
            }
        }
        
        // Test a/1 = a
        for a in 0..10 {
            assert_eq!(gf.divide(a, 1), a);
        }
        
        // Test 0/a = 0
        for a in 1..10 {
            assert_eq!(gf.divide(0, a), 0);
        }
    }
    
    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_division_by_zero() {
        let gf = GaloisField::new(0x11D);
        gf.divide(5, 0);
    }
    
    #[test]
    fn test_power() {
        let gf = GaloisField::new(0x11D);
        
        // a^0 = 1 for a != 0
        for a in 1..10 {
            assert_eq!(gf.power(a, 0), 1);
        }
        
        // a^1 = a
        for a in 0..10 {
            assert_eq!(gf.power(a, 1), a);
        }
        
        // a^n * a^m = a^(n+m)
        for a in 1..5 {
            for n in 1..5 {
                for m in 1..5 {
                    let lhs = gf.multiply(gf.power(a, n), gf.power(a, m));
                    let rhs = gf.power(a, n + m);
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }
    
    #[test]
    fn test_inverse() {
        let gf = GaloisField::new(0x11D);
        
        // a * a^(-1) = 1
        for a in 1..10 {
            let inv = gf.inverse(a);
            assert_eq!(gf.multiply(a, inv), 1);
        }
    }
    
    #[test]
    #[should_panic(expected = "Zero has no multiplicative inverse")]
    fn test_inverse_of_zero() {
        let gf = GaloisField::new(0x11D);
        gf.inverse(0);
    }
    
    #[test]
    fn test_polynomial_evaluation() {
        let gf = GaloisField::new(0x11D);
        
        // Test polynomial p(x) = x^2 + 2x + 3
        let poly = vec![1, 2, 3];
        
        // p(0) = 3
        assert_eq!(gf.polynomial_eval(&poly, 0), 3);
        
        // p(1) = 1 + 2 + 3 = 0 (XOR)
        assert_eq!(gf.polynomial_eval(&poly, 1), 0);
        
        // p(2) = 4 + 4 + 3 = 3
        assert_eq!(gf.polynomial_eval(&poly, 2), 3);
    }
    
    #[test]
    fn test_polynomial_batch_evaluation() {
        let gf = GaloisField::new(0x11D);
        
        // Test polynomials
        let poly1 = vec![1, 2, 3]; // x^2 + 2x + 3
        let poly2 = vec![4, 5, 6]; // 4x^2 + 5x + 6
        
        let polys = vec![poly1.clone(), poly2.clone()];
        let points = vec![0, 1, 2];
        
        let results = gf.polynomial_eval_batch(&polys, &points);
        
        // Check results for poly1
        assert_eq!(results[0][0], gf.polynomial_eval(&poly1, 0));
        assert_eq!(results[0][1], gf.polynomial_eval(&poly1, 1));
        assert_eq!(results[0][2], gf.polynomial_eval(&poly1, 2));
        
        // Check results for poly2
        assert_eq!(results[1][0], gf.polynomial_eval(&poly2, 0));
        assert_eq!(results[1][1], gf.polynomial_eval(&poly2, 1));
        assert_eq!(results[1][2], gf.polynomial_eval(&poly2, 2));
    }
    
    #[test]
    fn test_polynomial_addition() {
        let gf = GaloisField::new(0x11D);
        
        // (x^2 + 2x + 3) + (4x^2 + 5x + 6) = 5x^2 + 7x + 5
        let poly1 = vec![1, 2, 3];
        let poly2 = vec![4, 5, 6];
        let expected = vec![5, 7, 5];
        
        assert_eq!(gf.polynomial_add(&poly1, &poly2), expected);
        
        // Test different lengths
        let poly3 = vec![1, 2, 3, 4]; // x^3 + x^2 + 2x + 3
        let poly4 = vec![5, 6]; // 5x + 6
        let expected = vec![1, 2, 6, 6]; // x^3 + x^2 + 6x + 6
        
        assert_eq!(gf.polynomial_add(&poly3, &poly4), expected);
    }
    
    #[test]
    fn test_polynomial_multiplication() {
        let gf = GaloisField::new(0x11D);
        
        // (x + 1) * (x + 2) = x^2 + 3x + 2
        let poly1 = vec![1, 1];
        let poly2 = vec![1, 2];
        let expected = vec![1, 3, 2];
        
        assert_eq!(gf.polynomial_multiply(&poly1, &poly2), expected);
        
        // (x^2 + 1) * (x + 1) = x^3 + x^2 + x + 1
        let poly3 = vec![1, 0, 1];
        let poly4 = vec![1, 1];
        let expected = vec![1, 1, 1, 1];
        
        assert_eq!(gf.polynomial_multiply(&poly3, &poly4), expected);
    }
    
    #[test]
    fn test_syndrome_calculation() {
        let gf = GaloisField::new(0x11D);
        
        // Create a test message
        let message = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        // Calculate syndromes
        let syndrome_count = 4;
        let syndromes = gf.calculate_syndromes(&message, syndrome_count).unwrap();
        
        // Ensure we have the right number of syndromes
        assert_eq!(syndromes.len(), syndrome_count);
        
        // For a clean message with proper ECC, all syndromes would be zero
        // Since this is a test message without proper ECC, we just check they're calculated
        for &s in &syndromes {
            // Just ensure the calculation runs without error
            let _ = s;
        }
    }
}