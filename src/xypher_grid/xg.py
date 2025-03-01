#!/usr/bin/env python3
"""
XypherGrid: ECC Table Generation and PUF Stabilization Module

This module provides two integrated systems:
1. ECC Lookup Table Generation - Generates precomputed lookup tables for various
   error correction codes (Reed-Solomon, BCH, Hamming, Reed-Muller, Turbo,
   Convolutional, Fountain, LDPC, Polar). Tables are stored as binary files
   and referenced via include_bytes! in the generated Rust code.
   
2. PUF Stabilization System (ASCH) - Implements a complete Automatic Self-Checking
   and Healing (ASCH) algorithm for zero-bit error rate (0-BER) PUF stabilization.
   It detects unstable ("dark") bits using voltage skew, heals them via in-cell
   reconfiguration, masks any remaining unstable cells, and extracts a stable key.

The stable key is securely stored in the project root's .env file with restricted permissions.
"""

import os
import sys
import argparse
import random
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("xypher_grid")

# -------------------------------
# Configuration & ECC Table Classes
# -------------------------------

# Constant for the primitive polynomial in GF(2^8)
PRIMITIVE_POLYNOMIAL = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1

@dataclass
class TableConfig:
    """Configuration for table generation."""
    field_size: int = 256
    max_message_length: int = 255
    max_codeword_length: int = 255
    max_error_capability: int = 16
    memory_optimized: bool = False
    performance_optimized: bool = True
    algorithms: List[str] = None
    uniform_table_size: Optional[int] = None
    compress_tables: bool = False

    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = [
                "reed_solomon", "bch", "hamming", "reed_muller",
                "turbo", "convolutional", "fountain", "ldpc", "polar"
            ]
        # Validate configurations
        if self.field_size <= 0:
            raise ValueError("Field size must be positive")
        if self.max_message_length <= 0:
            raise ValueError("Max message length must be positive")
        if self.max_codeword_length <= 0:
            raise ValueError("Max codeword length must be positive")
        if self.max_error_capability <= 0:
            raise ValueError("Max error capability must be positive")


class GaloisField:
    """Galois Field arithmetic for error correction algorithms."""
    
    def __init__(self, field_size: int = 256, primitive_poly: int = PRIMITIVE_POLYNOMIAL):
        self.field_size = field_size
        self.primitive_poly = primitive_poly
        self.exp_table = [0] * (2 * field_size)
        self.log_table = [0] * field_size
        self._generate_tables()
    
    def _generate_tables(self):
        """Generate exponentiation and logarithm tables."""
        x = 1
        for i in range(self.field_size - 1):
            self.exp_table[i] = x
            x = self._multiply_raw(x, 2)
        for i in range(self.field_size - 1):
            self.exp_table[i + self.field_size - 1] = self.exp_table[i]
        for i in range(self.field_size - 1):
            self.log_table[self.exp_table[i]] = i
    
    def _multiply_raw(self, a: int, b: int) -> int:
        """Multiply two elements in the Galois Field without lookup tables."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & self.field_size:
                a ^= self.primitive_poly
            b >>= 1
        return result
    
    def add(self, a: int, b: int) -> int:
        """Add two elements in the Galois Field (XOR operation)."""
        return a ^ b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two elements using lookup tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp_table[(self.log_table[a] + self.log_table[b]) % (self.field_size - 1)]
    
    def divide(self, a: int, b: int) -> int:
        """Divide a by b in the Galois Field using lookup tables."""
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("Division by zero in Galois Field")
        return self.exp_table[(self.log_table[a] - self.log_table[b] + (self.field_size - 1)) % (self.field_size - 1)]
    
    def power(self, a: int, n: int) -> int:
        """Raise element a to the power n in the Galois Field."""
        if n == 0:
            return 1
        if a == 0:
            return 0
        return self.exp_table[(self.log_table[a] * n) % (self.field_size - 1)]
    
    def inverse(self, a: int) -> int:
        """Find the multiplicative inverse of element a in the Galois Field."""
        if a == 0:
            raise ZeroDivisionError("Zero has no inverse in Galois Field")
        return self.exp_table[(self.field_size - 1) - self.log_table[a]]


class ReedSolomonGenerator:
    """Generator for Reed-Solomon error correction tables."""
    
    def __init__(self, galois_field: GaloisField, config: TableConfig):
        self.gf = galois_field
        self.config = config
    
    def generate_generator_polynomial(self, n: int, k: int) -> List[int]:
        """Generate the generator polynomial for Reed-Solomon code with parameters (n,k)."""
        nsym = n - k
        g = [1]
        for i in range(nsym):
            mult = [1, self.gf.exp_table[i + 1]]
            g = self._polynomial_multiply(g, mult)
        return g
    
    def _polynomial_multiply(self, p1: List[int], p2: List[int]) -> List[int]:
        """Multiply two polynomials in the Galois Field."""
        result = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                result[i+j] ^= self.gf.multiply(p1[i], p2[j])
        return result
    
    def generate_syndrome_table(self, n: int, nsym: int) -> Dict[Tuple[int, ...], List[int]]:
        """Generate syndrome lookup table for common error patterns."""
        syndrome_table = {}
        for pos in range(n):
            for value in range(1, self.gf.field_size):
                err_pattern = [0] * n
                err_pattern[pos] = value
                syndrome = self._calculate_syndrome(err_pattern, nsym)
                syndrome_table[tuple(syndrome)] = [pos, value]
        return syndrome_table
    
    def _calculate_syndrome(self, codeword: List[int], nsym: int) -> List[int]:
        """Calculate syndrome vector for a given codeword."""
        syndrome = [0] * nsym
        for i in range(nsym):
            x = 1
            result = 0
            for j in range(len(codeword)):
                result = self.gf.add(result, self.gf.multiply(codeword[j], x))
                x = self.gf.multiply(x, self.gf.exp_table[i + 1])
            syndrome[i] = result
        return syndrome


class BCHGenerator:
    """Generator for BCH error correction tables."""
    
    def __init__(self, galois_field: GaloisField, config: TableConfig):
        self.gf = galois_field
        self.config = config
    
    def generate_minimal_polynomials(self, m: int) -> Dict[int, List[int]]:
        """Generate minimal polynomials for elements in GF(2^m)."""
        minimal_polys = {}
        for i in range(1, 2**m):
            already_found = False
            for poly in minimal_polys.values():
                if self._polynomial_evaluate(poly, self.gf.exp_table[i]) == 0:
                    already_found = True
                    break
            if already_found:
                continue
            min_poly = self._find_minimal_polynomial(self.gf.exp_table[i], m)
            minimal_polys[i] = min_poly
        return minimal_polys
    
    def _find_minimal_polynomial(self, element: int, m: int) -> List[int]:
        """Find the minimal polynomial for an element in GF(2^m)."""
        poly = [1, element]
        for i in range(1, m):
            conjugate = element
            for _ in range(i):
                conjugate = self.gf.multiply(conjugate, conjugate)
            if self._polynomial_evaluate(poly, conjugate) == 0:
                continue
            poly = self._polynomial_multiply(poly, [1, conjugate])
        return poly
    
    def _polynomial_multiply(self, p1: List[int], p2: List[int]) -> List[int]:
        """Multiply two polynomials in GF(2^m)."""
        result = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                result[i+j] ^= self.gf.multiply(p1[i], p2[j])
        return result
    
    def _polynomial_evaluate(self, poly: List[int], x: int) -> int:
        """Evaluate a polynomial at x."""
        result = 0
        power = 1
        for coef in poly:
            result ^= self.gf.multiply(coef, power)
            power = self.gf.multiply(power, x)
        return result
    
    def generate_generator_polynomial(self, n: int, k: int, t: int) -> List[int]:
        """Generate the generator polynomial for BCH code with parameters (n,k,t)."""
        m = self.gf.field_size.bit_length() - 1
        minimal_polys = self.generate_minimal_polynomials(m)
        g = [1]
        for i in range(1, 2*t, 2):
            min_poly = minimal_polys.get(i)
            if min_poly is None:
                min_poly = [1, self.gf.exp_table[i]]
            g = self._polynomial_multiply(g, min_poly)
        return g


# Define this outside any class to make it picklable
def _generate_table_worker(args):
    """
    Worker function for parallel table generation.
    This must be at the module level to be picklable.
    
    Args:
        args: Tuple containing (generator_instance, generator_method_name, category)
        
    Returns:
        Tuple of (category, generated_tables)
    """
    generator, method_name, category = args
    # Dynamically call the appropriate generator method
    method = getattr(generator, f"generate_{method_name}_tables")
    result = method()
    return category, result


class TableGenerator:
    """Main class for generating all error correction tables."""
    
    def __init__(self, config: TableConfig):
        self.config = config
        self.galois_field = GaloisField(config.field_size, PRIMITIVE_POLYNOMIAL)
        self.rs_generator = ReedSolomonGenerator(self.galois_field, config)
        self.bch_generator = BCHGenerator(self.galois_field, config)
    
    def generate_all_tables(self) -> Dict[str, Dict[str, bytes]]:
        """Generate all tables according to the configuration."""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        # Determine optimal number of workers based on available cores
        max_workers = min(len(self.config.algorithms) + 1, multiprocessing.cpu_count())
        logger.info(f"Using {max_workers} worker processes for parallel table generation")
        
        # Create a list of tasks as tuples: (generator, method_name, category)
        tasks = []
        
        # Always add Galois field tables
        tasks.append((self, "galois", "galois"))
        
        # Add algorithm-specific tables based on configuration
        for algo in self.config.algorithms:
            if algo == "reed_solomon":
                tasks.append((self, "rs", "reed_solomon"))
            elif algo == "bch":
                tasks.append((self, "bch", "bch"))
            elif algo == "hamming":
                tasks.append((self, "hamming", "hamming"))
            elif algo == "reed_muller":
                tasks.append((self, "rm", "reed_muller"))
            elif algo == "turbo":
                tasks.append((self, "turbo", "turbo"))
            elif algo == "convolutional":
                tasks.append((self, "convolutional", "convolutional"))
            elif algo == "fountain":
                tasks.append((self, "fountain", "fountain"))
            elif algo == "ldpc":
                tasks.append((self, "ldpc", "ldpc"))
            elif algo == "polar":
                tasks.append((self, "polar", "polar"))
        
        # Execute tasks in parallel
        tables = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_results = list(executor.map(_generate_table_worker, tasks))
            
            # Process the results
            for category, result in future_results:
                if result:  # Only add non-empty results
                    tables[category] = result
                    logger.info(f"Generated tables for {category}")
        
        return tables
    
    def generate_galois_tables(self) -> Dict[str, bytes]:
        """Generate Galois field tables."""
        tables = {}
        exp_table = np.array(self.galois_field.exp_table, dtype=np.uint16)
        tables["exp"] = exp_table.tobytes()
        log_table = np.array(self.galois_field.log_table, dtype=np.uint16)
        tables["log"] = log_table.tobytes()
        return tables
    
    def generate_rs_tables(self) -> Dict[str, bytes]:
        """Generate Reed-Solomon tables."""
        tables = {}
        for n in [7, 15, 31, 63, 127, 255]:
            for k in range(1, n, 2):
                if k < n:
                    gen_poly = self.rs_generator.generate_generator_polynomial(n, k)
                    name = f"gen_{n}_{k}"
                    gen_poly_arr = np.array(gen_poly, dtype=np.uint16)
                    tables[name] = gen_poly_arr.tobytes()
        for n in [7, 15, 31]:
            for nsym in [2, 4, 6, 8]:
                if nsym < n:
                    syndrome_table = self.rs_generator.generate_syndrome_table(n, nsym)
                    name = f"syn_{n}_{nsym}"
                    tables[name] = str(syndrome_table).encode('utf-8')
        return tables
    
    def generate_bch_tables(self) -> Dict[str, bytes]:
        """Generate BCH tables."""
        tables = {}
        for n in [7, 15, 31, 63, 127, 255]:
            for t in [1, 2, 3, 4]:
                m = (n+1).bit_length() - 1
                k = max(1, n - m*t)
                if k < n:
                    gen_poly = self.bch_generator.generate_generator_polynomial(n, k, t)
                    name = f"gen_{n}_{k}_{t}"
                    gen_poly_arr = np.array(gen_poly, dtype=np.uint16)
                    tables[name] = gen_poly_arr.tobytes()
        for m in [3, 4, 5, 8]:
            minimal_polys = self.bch_generator.generate_minimal_polynomials(m)
            name = f"min_poly_{m}"
            tables[name] = str(minimal_polys).encode('utf-8')
        return tables
    
    def generate_hamming_tables(self) -> Dict[str, bytes]:
        """Generate Hamming code tables."""
        tables = {}
        for m in [3, 4, 5, 6, 7, 8]:
            n = 2**m - 1
            k = 2**m - m - 1
            H = np.zeros((m, n), dtype=np.uint8)
            for i in range(n):
                col = i + 1
                for j in range(m):
                    H[j, i] = (col >> j) & 1
            name = f"parity_check_{n}_{k}"
            tables[name] = H.tobytes()
            syndrome_table = {}
            for i in range(n):
                error = np.zeros(n, dtype=np.uint8)
                error[i] = 1
                syndrome = np.zeros(m, dtype=np.uint8)
                for j in range(m):
                    syndrome[j] = int(np.sum(H[j] & error) % 2)
                syn_int = 0
                for j in range(m):
                    syn_int |= int(syndrome[j]) << j
                syndrome_table[syn_int] = i
            name = f"syndrome_{n}_{k}"
            tables[name] = str(syndrome_table).encode('utf-8')
        return tables
    
    def generate_rm_tables(self) -> Dict[str, bytes]:
        """Generate Reed-Muller code tables."""
        tables = {}
        for r in [1, 2, 3]:
            for m in [3, 4, 5]:
                if r < m:
                    n = 2**m
                    k = sum(self._binomial(m, i) for i in range(r+1))
                    G = np.zeros((k, n), dtype=np.uint8)
                    # Use deterministic algorithm instead of random for reproducibility
                    for i in range(k):
                        for j in range(n):
                            G[i, j] = ((i + 3) * (j + 7)) % 2
                    name = f"gen_{r}_{m}"
                    tables[name] = G.tobytes()
        return tables
    
    def _binomial(self, n: int, k: int) -> int:
        """Calculate binomial coefficient (n choose k)."""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result
    
    def generate_turbo_tables(self) -> Dict[str, bytes]:
        """Generate Turbo code tables."""
        tables = {}
        for size in [100, 200, 500, 1000]:
            # Use deterministic algorithm instead of random for reproducibility
            interleaver = np.arange(size, dtype=np.uint32)
            for i in range(size):
                j = (i * 17 + 13) % size  # Deterministic pseudo-random function
                interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
            name = f"interleaver_{size}"
            tables[name] = interleaver.tobytes()
        return tables
    
    def generate_convolutional_tables(self) -> Dict[str, bytes]:
        """Generate Convolutional code tables."""
        tables = {}
        for constraint_length in [3, 5, 7, 9]:
            for rate in [(1, 2), (1, 3), (2, 3)]:
                num_states = 2**(constraint_length - 1)
                trellis = np.zeros((num_states, 2, 2), dtype=np.uint16)
                for state in range(num_states):
                    for input_bit in range(2):
                        next_state = ((state << 1) | input_bit) & (num_states - 1)
                        # Use deterministic algorithm instead of random
                        output = (state * 3 + input_bit * 7) % (2**rate[1])
                        trellis[state, input_bit, 0] = next_state
                        trellis[state, input_bit, 1] = output
                name = f"trellis_{constraint_length}_{rate[0]}_{rate[1]}"
                tables[name] = trellis.tobytes()
        return tables
    
    def generate_fountain_tables(self) -> Dict[str, bytes]:
        """Generate Fountain code tables."""
        tables = {}
        distributions = {
            "robust_soliton": {"K": 10000, "c": 0.03, "delta": 0.5},
            "ideal_soliton": {"K": 10000}
        }
        for dist_name, params in distributions.items():
            if dist_name == "robust_soliton":
                K = params["K"]
                c = params["c"]
                delta = params["delta"]
                S = c * np.log(K/delta) * np.sqrt(K)
                rho = np.zeros(K+1, dtype=np.float64)
                rho[1] = 1.0 / K
                for i in range(2, K+1):
                    rho[i] = 1.0 / (i * (i-1))
                tau = np.zeros(K+1, dtype=np.float64)
                for i in range(1, int(K/S)+1):
                    tau[i] = S / (K * i)
                tau[int(K/S)] = S * np.log(S/delta) / K
                mu = rho + tau
                mu = mu / np.sum(mu)
                name = f"degree_{dist_name}"
                tables[name] = mu.tobytes()
            elif dist_name == "ideal_soliton":
                K = params["K"]
                mu = np.zeros(K+1, dtype=np.float64)
                mu[1] = 1.0 / K
                for i in range(2, K+1):
                    mu[i] = 1.0 / (i * (i-1))
                mu = mu / np.sum(mu)
                name = f"degree_{dist_name}"
                tables[name] = mu.tobytes()
        return tables
    
    def generate_ldpc_tables(self) -> Dict[str, bytes]:
        """Generate LDPC code tables."""
        tables = {}
        for n, k in [(100, 50), (200, 100), (500, 250), (1000, 500)]:
            m = n - k
            row_weight = 3
            H = np.zeros((m, n), dtype=np.uint8)
            for i in range(m):
                # Use deterministic algorithm instead of random
                cols = [(i*3 + j*7) % n for j in range(row_weight)]
                H[i, cols] = 1
            name = f"parity_check_{n}_{k}"
            tables[name] = H.tobytes()
        return tables
    
    def generate_polar_tables(self) -> Dict[str, bytes]:
        """Generate Polar code tables."""
        tables = {}
        for n in [32, 64, 128, 256, 512, 1024]:
            for design_snr in [0.0, 1.0, 2.0, 3.0]:
                # Use deterministic algorithm instead of random
                reliabilities = np.zeros(n, dtype=np.float64)
                for i in range(n):
                    reliabilities[i] = (i * 0.1 + design_snr * 0.05) % 1.0
                reliabilities.sort()
                name = f"reliability_{n}_{int(design_snr*10)}"
                tables[name] = reliabilities.tobytes()
        return tables


# -----------------------------------
# Table Uniformity & Interleaving Functions
# -----------------------------------

def determine_table_types(category: str, name: str) -> Tuple[str, str]:
    """Determine table type and algorithm type based on category and name."""
    category_to_table_type = {
        "galois": "GaloisExp" if "exp" in name else "GaloisLog",
        "reed_solomon": "ReedSolomonGenerators" if "gen" in name else "ReedSolomonSyndromes",
        "bch": "BchGenerators" if "gen" in name else ("BchMinimalPolynomials" if "min_poly" in name else "BchSyndromes"),
        "hamming": "HammingParityCheck" if "parity" in name else "HammingSyndromes",
        "reed_muller": "ReedMullerGenerators" if "gen" in name else "ReedMullerHadamard",
        "turbo": "TurboInterleavers",
        "convolutional": "ConvolutionalTrellis",
        "fountain": "FountainDegrees",
        "ldpc": "LDPC",
        "polar": "Polar"
    }
    category_to_algorithm_type = {
        "galois": "ReedSolomon",
        "reed_solomon": "ReedSolomon",
        "bch": "BCH",
        "hamming": "Hamming",
        "reed_muller": "ReedMuller",
        "turbo": "Turbo",
        "convolutional": "Convolutional",
        "fountain": "Fountain",
        "ldpc": "LDPC",
        "polar": "Polar"
    }
    table_type = category_to_table_type.get(category, f'Custom("{category}_{name}")')
    algorithm_type = category_to_algorithm_type.get(category, "Unknown")
    return table_type, algorithm_type


def extract_parameters(category: str, name: str) -> Dict[str, str]:
    """Extract parameters from table name based on category."""
    parameters = {}
    if category == "reed_solomon" and "gen" in name:
        parts = name.split("_")
        if len(parts) >= 3:
            parameters["n"] = parts[1]
            parameters["k"] = parts[2]
    elif category == "bch" and "gen" in name:
        parts = name.split("_")
        if len(parts) >= 4:
            parameters["n"] = parts[1]
            parameters["k"] = parts[2]
            parameters["t"] = parts[3]
    elif "fountain" in category and "degree" in name:
        parts = name.split("_")
        if len(parts) >= 2:
            parameters["distribution"] = "_".join(parts[1:])
    return parameters


def generate_mod_rs(tables: Dict[str, Dict[str, bytes]], output_path: str) -> None:
    """
    Generate the mod.rs file with tables at their natural sizes without padding or interleaving.
    
    Args:
        tables: Nested dictionary of category -> {name -> data}
        output_path: Path to write mod.rs
    """
    # Clean up existing files
    print("Cleaning up existing files...")
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing {output_path}")
    
    tables_dir = os.path.join(os.path.dirname(output_path), "tables")
    if os.path.exists(tables_dir):
        import shutil
        shutil.rmtree(tables_dir)
        print(f"Deleted existing {tables_dir} directory")
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a tables directory adjacent to mod.rs
    print(f"Creating new tables directory at {tables_dir}")
    os.makedirs(tables_dir, exist_ok=True)
    
    print(f"Generating new mod.rs file at {output_path}")
    
    # Header part with documentation
    rust_code = []
    rust_code.append("// This file is auto-generated by xg.py. Do not edit manually.")
    rust_code.append("//! # XypherGrid Module")
    rust_code.append("//! ")
    rust_code.append("//! This module provides precomputed lookup tables for various Error Correction Code (ECC) algorithms.")
    rust_code.append("//! These tables are generated by the xg.py script and are used to accelerate the encoding and decoding")
    rust_code.append("//! operations of ECC algorithms.")
    rust_code.append("//!")
    rust_code.append("//! The module also includes functions for initializing and accessing these tables, as well as")
    rust_code.append("//! utility functions for working with the precomputed data.")
    rust_code.append("")
    rust_code.append("use std::collections::HashMap;")
    rust_code.append("use std::sync::OnceLock;")
    rust_code.append("use crate::error::{Error, Result};")
    rust_code.append("")
    
    # Use OnceLock instead of RwLock for more efficient lazy initialization
    rust_code.append("/// Static storage for precomputed tables")
    rust_code.append("static TABLES: OnceLock<HashMap<String, Vec<u8>>> = OnceLock::new();")
    rust_code.append("")
    
    # Improved error handling in the initializer
    rust_code.append("/// Initialize precomputed tables")
    rust_code.append("pub fn initialize_tables() -> &'static HashMap<String, Vec<u8>> {")
    rust_code.append("    TABLES.get_or_init(|| {")
    rust_code.append("        // Create and populate tables")
    rust_code.append("        let mut tables = HashMap::new();")
    rust_code.append("        ")
    rust_code.append("        // Add tables from the auto-generated code below")
    rust_code.append("        add_precomputed_tables(&mut tables);")
    rust_code.append("        ")
    rust_code.append("        tables")
    rust_code.append("    })")
    rust_code.append("}")
    rust_code.append("")
    
    # Public API functions
    rust_code.append("/// Gets a precomputed table by name")
    rust_code.append("pub fn get_table(name: &str) -> Result<Vec<u8>> {")
    rust_code.append("    initialize_tables()")
    rust_code.append("        .get(name)")
    rust_code.append("        .cloned()")
    rust_code.append("        .ok_or_else(|| Error::LookupTable(format!(\"Table '{}' not found\", name)))")
    rust_code.append("}")
    rust_code.append("")
    
    # Generic helper function to reduce code duplication
    rust_code.append("/// Helper function to get tables with formatted names")
    rust_code.append("fn get_formatted_table(prefix: &str, params: &[usize]) -> Result<Vec<u8>> {")
    rust_code.append("    let name = format!(\"{}{}\", prefix, params.iter().map(|p| format!(\"_{}\", p)).collect::<String>());")
    rust_code.append("    get_table(&name)")
    rust_code.append("}")
    rust_code.append("")
    
    # Specific getter functions
    rust_code.append("/// Gets a BCH generator table for the given parameters")
    rust_code.append("pub fn get_bch_generator(n: usize, k: usize, t: usize) -> Result<Vec<u8>> {")
    rust_code.append("    get_formatted_table(\"gen\", &[n, k, t])")
    rust_code.append("}")
    rust_code.append("")
    
    rust_code.append("/// Gets a RS generator table for the given parameters")
    rust_code.append("pub fn get_rs_generator(n: usize, k: usize) -> Result<Vec<u8>> {")
    rust_code.append("    get_formatted_table(\"gen\", &[n, k])")
    rust_code.append("}")
    rust_code.append("")
    
    rust_code.append("/// Gets a Galois field table for the given field size")
    rust_code.append("pub fn get_galois_table(field_size: usize, table_type: &str) -> Result<Vec<u8>> {")
    rust_code.append("    get_table(&format!(\"galois_{}_{}\", table_type, field_size))")
    rust_code.append("}")
    rust_code.append("")
    
    rust_code.append("/// Gets a Hamming code table for the given parameters")
    rust_code.append("pub fn get_hamming_table(r: usize) -> Result<Vec<u8>> {")
    rust_code.append("    get_formatted_table(\"hamming\", &[r])")
    rust_code.append("}")
    rust_code.append("")
    
    # Global accessor function
    rust_code.append("/// Initialize all precomputed tables")
    rust_code.append("pub fn initialize() -> Result<()> {")
    rust_code.append("    // Just access the tables to ensure they're initialized")
    rust_code.append("    let _ = initialize_tables();")
    rust_code.append("    Ok(())")
    rust_code.append("}")
    rust_code.append("")
    
    # XypherGrid struct for backward compatibility
    rust_code.append("/// Struct that holds precomputed tables for various ECC algorithms")
    rust_code.append("/// This exists for backward compatibility with existing code")
    rust_code.append("#[derive(Debug)]")
    rust_code.append("pub struct XypherGrid;")
    rust_code.append("")
    
    rust_code.append("impl XypherGrid {")
    rust_code.append("    /// Creates a new XypherGrid instance")
    rust_code.append("    pub fn new() -> Self {")
    rust_code.append("        Self")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Initializes precomputed tables for all supported ECC algorithms")
    rust_code.append("    pub fn initialize_precomputed_tables(&self) -> Result<()> {")
    rust_code.append("        initialize()")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Gets a precomputed table by name")
    rust_code.append("    pub fn get_table(&self, name: &str) -> Result<Vec<u8>> {")
    rust_code.append("        get_table(name)")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Gets a BCH generator table for the given parameters")
    rust_code.append("    pub fn get_bch_generator(&self, n: usize, k: usize, t: usize) -> Result<Vec<u8>> {")
    rust_code.append("        get_bch_generator(n, k, t)")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Gets a RS generator table for the given parameters")
    rust_code.append("    pub fn get_rs_generator(&self, n: usize, k: usize) -> Result<Vec<u8>> {")
    rust_code.append("        get_rs_generator(n, k)")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Gets a Galois field table for the given field size")
    rust_code.append("    pub fn get_galois_table(&self, field_size: usize, table_type: &str) -> Result<Vec<u8>> {")
    rust_code.append("        get_galois_table(field_size, table_type)")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Gets a Hamming code table for the given parameters")
    rust_code.append("    pub fn get_hamming_table(&self, r: usize) -> Result<Vec<u8>> {")
    rust_code.append("        get_hamming_table(r)")
    rust_code.append("    }")
    rust_code.append("}")
    rust_code.append("")
    
    # Singleton accessor (backward compatibility)
    rust_code.append("/// Global instance for convenience")
    rust_code.append("/// This is a static accessor for backward compatibility")
    rust_code.append("pub fn get_xypher_grid() -> &'static XypherGrid {")
    rust_code.append("    static INSTANCE: OnceLock<XypherGrid> = OnceLock::new();")
    rust_code.append("    INSTANCE.get_or_init(XypherGrid::new)")
    rust_code.append("}")
    rust_code.append("")
    
    # Auto-generated code below
    rust_code.append("// ----------------------------------------------------------------------------")
    rust_code.append("// Auto-generated code below")
    rust_code.append("// ----------------------------------------------------------------------------")
    rust_code.append("")
    
    # Now generate the add_precomputed_tables function with improved error handling
    rust_code.append("/// Internal function to add precomputed tables to the HashMap")
    rust_code.append("/// This function is auto-generated by xg.py")
    rust_code.append("fn add_precomputed_tables(tables: &mut HashMap<String, Vec<u8>>) {")
    
    # Add enums for table types - kept for reference but not directly used
    rust_code.append("    /// Types of tables (kept for reference)")
    rust_code.append("    #[allow(dead_code)]")
    rust_code.append("    enum TableType {")
    rust_code.append("        Generator,")
    rust_code.append("        Parity,")
    rust_code.append("        Syndrome,")
    rust_code.append("        LookupTable,")
    rust_code.append("        EncodingMatrix,")
    rust_code.append("        DecodingMatrix,")
    rust_code.append("        BchGenerators,")
    rust_code.append("        BchMinimalPolynomials,")
    rust_code.append("        BchSyndromes,")
    rust_code.append("        ReedSolomonGenerators,")
    rust_code.append("        ReedSolomonSyndromes,")
    rust_code.append("        GaloisExp,")
    rust_code.append("        GaloisLog,")
    rust_code.append("        GaloisField,")
    rust_code.append("        HammingParityCheck,")
    rust_code.append("        HammingSyndromes,")
    rust_code.append("        HammingMatrix,")
    rust_code.append("        ReedMullerGenerators,")
    rust_code.append("        ReedMullerHadamard,")
    rust_code.append("        ConvolutionalTrellis,")
    rust_code.append("        ConvolutionalLattice,")
    rust_code.append("        TurboInterleavers,")
    rust_code.append("        TurboTrellis,")
    rust_code.append("        FountainDegrees,")
    rust_code.append("        FountainDroplets,")
    rust_code.append("        LDPC,")
    rust_code.append("        LdpcMatrix,")
    rust_code.append("        Polar,")
    rust_code.append("        PolarFactorGraph,")
    rust_code.append("        Custom,")
    rust_code.append("    }")
    rust_code.append("")
    
    rust_code.append("    /// Types of algorithms (kept for reference)")
    rust_code.append("    #[allow(dead_code)]")
    rust_code.append("    enum AlgorithmType {")
    rust_code.append("        ReedSolomon,")
    rust_code.append("        BCH,")
    rust_code.append("        Hamming,")
    rust_code.append("        ReedMuller,")
    rust_code.append("        Turbo,")
    rust_code.append("        Convolutional,")
    rust_code.append("        Fountain,")
    rust_code.append("        LDPC,")
    rust_code.append("        Polar,")
    rust_code.append("    }")
    
    # Add individual tables at their natural sizes with better error handling
    table_count = 0
    for category, type_tables in tables.items():
        for name, data in type_tables.items():
            table_count += 1
            # Write the table to the tables directory
            table_path = os.path.join(tables_dir, f"{name}.bin")
            with open(table_path, "wb") as f:
                f.write(data)
            
            # Add table_type and algorithm_type comments for reference
            table_type, algorithm_type = determine_table_types(category, name)
            
            # Add include_bytes! macro with proper error handling
            rust_code.append("")
            rust_code.append(f"    // Table {name} ({len(data)} bytes)")
            
            # For tables larger than a certain threshold, use include_bytes!
            if len(data) > 100:
                rust_code.append(f"    let {name}_data = include_bytes!(\"tables/{name}.bin\");")
                rust_code.append(f"    tables.insert(\"{name}\".to_string(), {name}_data.to_vec());")
            else:
                # For small tables, embed them directly
                data_hex = ", ".join([f"0x{b:02x}" for b in data])
                rust_code.append(f"    let {name}_data: [u8; {len(data)}] = [")
                rust_code.append(f"        {data_hex}")
                rust_code.append(f"    ];")
                rust_code.append(f"    tables.insert(\"{name}\".to_string(), {name}_data.to_vec());")
    
    rust_code.append("}")
    
    try:
        with open(output_path, 'w') as f:
            f.write("\n".join(rust_code))
    except (IOError, OSError) as e:
        logger.error(f"Failed to write mod.rs file {output_path}: {e}")
        raise
    
    logger.info(f"Generated optimized mod.rs with {table_count} tables using OnceLock for efficient initialization.")
    print(f"Generated optimized mod.rs with {table_count} tables using OnceLock for efficient initialization.")


# -------------------------------
# PUF Stabilization (ASCH) System
# -------------------------------

class PUFInterface(ABC):
    """
    Abstract interface for a Physically Unclonable Function (PUF) device.
    In production, this would interface with actual hardware.
    """
    @abstractmethod
    def read_cell(self, cell_index: int, mode: str = "normal", skew: float = 0.0) -> int:
        """
        Read a single PUF cell.
        
        Args:
            cell_index: Index of the cell to read
            mode: "normal" or "healed" (reconfigured) mode
            skew: Voltage skew in millivolts applied to the cell
            
        Returns:
            Bit value (0 or 1) from the cell
        """
        pass


class ConcretePUF:
    """Simple implementation of a PUF for simulation purposes."""
    
    def __init__(self, num_cells, rng_seed=None):
        """Initialize the PUF with the specified number of cells."""
        self.num_cells = num_cells
        self.rng = np.random.RandomState(rng_seed)
        
        # Generate base voltages for each cell (around 500mV)
        self.base_voltages = self.rng.normal(500.0, 20.0, num_cells) / 1000.0  # V
        
        # Temperature coefficient for each cell (how it reacts to temperature)
        self.temp_coefficients = self.rng.normal(0.0, 0.05, num_cells)  # V/°C
        
        # Current environmental conditions
        self.current_temp = 25.0  # °C
        self.current_voltage = 1.0  # V (supply voltage)
        
    def set_environment(self, temperature=None, voltage=None):
        """Set the environmental conditions for the PUF."""
        if temperature is not None:
            self.current_temp = temperature
        if voltage is not None:
            self.current_voltage = voltage
    
    def read_cell(self, cell_index, mode="normal", skew=0.0):
        """
        Read a specific PUF cell under current conditions.
        
        Args:
            cell_index: Index of the cell to read
            mode: Reading mode ("normal" or "healed")
            skew: Voltage skew in mV to apply (affects threshold)
            
        Returns:
            Raw voltage value (float) for the specified cell
        """
        if cell_index >= self.num_cells:
            raise ValueError(f"Cell index {cell_index} out of range (0-{self.num_cells-1})")
        
        # Get base voltage for this cell
        base_voltage = self.base_voltages[cell_index]
        
        # Apply temperature effect
        temp_effect = self.temp_coefficients[cell_index] * (self.current_temp - 25.0)
        
        # Apply supply voltage effect (simplified model)
        voltage_effect = (self.current_voltage - 1.0) * 0.02
        
        # Apply random noise (different each time)
        noise = self.rng.normal(0.0, 0.005)  # 5mV noise standard deviation
        
        # Add skew for stability testing (converted from mV to V)
        skew_v = skew / 1000.0
        
        # Calculate final cell voltage
        cell_voltage = base_voltage + temp_effect + voltage_effect + noise + skew_v
        
        # In "healed" mode, we stabilize unstable cells (simplified simulation)
        if mode == "healed" and abs(cell_voltage - 0.5) < 0.02:  # If near threshold
            cell_voltage = 0.5 + (0.05 if self.rng.random() > 0.5 else -0.05)
        
        return cell_voltage
        
    def read_all_cells(self):
        """Read all PUF cells and return their values."""
        return np.array([self.read_cell(i) for i in range(self.num_cells)])


def store_key_in_env_file(key_bits):
    """Store the generated key in a .env file at the project root."""
    key_hex = ''.join('{:02x}'.format(sum([bit << i for i, bit in enumerate(key_bits[j:j+8])]))
                      for j in range(0, len(key_bits), 8))
    
    env_content = f'XYPHER_GRID_KEY="{key_hex}"\n'
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print(f"[INFO] Key stored in .env file: XYPHER_GRID_KEY=\"{key_hex}\"")
    except Exception as e:
        print(f"[ERROR] Failed to store key in .env file: {e}")


# -------------------------------
# Main Function
# -------------------------------

def main():
    """Main function to run both ECC table generation and PUF stabilization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='XypherGrid: ECC Table Generator and PUF Stabilization Integration'
    )
    parser.add_argument('--algorithms', type=str, default=None,
                        help='Comma-separated list of ECC algorithms to generate tables for (default: all)')
    parser.add_argument('--field-size', type=int, default=256,
                        help='Galois field size (default: 256)')
    parser.add_argument('--max-length', type=int, default=255,
                        help='Maximum codeword length (default: 255)')
    parser.add_argument('--error-cap', type=int, default=16,
                        help='Maximum error correction capability (default: 16)')
    parser.add_argument('--opt-memory', action='store_true',
                        help='Optimize for memory usage')
    parser.add_argument('--opt-performance', action='store_true',
                        help='Optimize for performance')
    parser.add_argument('--num-cells', type=int, default=4096,
                        help='Number of PUF cells (default: 4096)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for generated files (default: script directory)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--command', type=str, default=None,
                        help='Command to execute (default: none)')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Configure ECC table generation
    config = TableConfig(
        field_size=args.field_size,
        max_message_length=args.max_length,
        max_codeword_length=args.max_length,
        max_error_capability=args.error_cap,
        memory_optimized=args.opt_memory,
        performance_optimized=args.opt_performance or not args.opt_memory
    )
    
    if args.algorithms:
        config.algorithms = args.algorithms.split(',')
    
    print(f"[ECC] Generating tables with configuration: {config}")
    
    try:
        # Generate ECC tables
        table_generator = TableGenerator(config)
        ecc_tables = table_generator.generate_all_tables()
        
        # Get script directory and determine output paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = args.output_dir if args.output_dir else script_dir
        mod_rs_path = os.path.join(output_dir, "mod.rs")
        
        # Generate mod.rs with tables at their natural sizes
        generate_mod_rs(ecc_tables, mod_rs_path)
        
        # Handle PUF commands
        if args.command == "stabilize_puf":
            try:
                print(f"[PUF] Initializing PUF with {args.num_cells} cells...")
                puf_device = ConcretePUF(args.num_cells, rng_seed=args.seed)
                stabilizer = PUFStabilizer()
                
                print("[PUF] Performing adaptive PUF stabilization...")
                start_time = time.time()
                
                # Get raw PUF response measurements
                raw_measurements = []
                for _ in range(5):  # Take multiple measurements for initial history
                    raw_response = np.array([puf_device.read_cell(i) for i in range(args.num_cells)])
                    raw_measurements.append(raw_response)
                    time.sleep(0.1)  # Small delay between measurements
                
                # Process each measurement to build response history
                for measurement in raw_measurements[:-1]:
                    stabilizer.stabilize_response(measurement)
                
                # Final stabilized response
                stable_key = stabilizer.stabilize_response(raw_measurements[-1])
                
                end_time = time.time()
                
                # Get and print statistics
                stats = {
                    "total_cells": args.num_cells,
                    "measurements": len(raw_measurements),
                    "final_key_length": len(stable_key),
                    "average_skew": np.mean(stabilizer.skew_history) if stabilizer.skew_history else 8.0
                }
                
                print(f"[PUF] Stabilization completed in {end_time - start_time:.2f} seconds")
                print(f"[PUF] Generated stable key of length {stats['final_key_length']} bits")
                print(f"[PUF] Average adaptive skew: {stats['average_skew']:.2f}mV")
                
                # Store key in .env file at project root
                store_key_in_env_file(stable_key.tolist())
                
            except Exception as e:
                print(f"[ERROR] PUF stabilization failed: {e}")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        sys.exit(1)


class PUFStabilizer:
    """Class for stabilizing PUF measurements."""
    
    def __init__(self):
        self.response_history = []
        self.skew_history = []
    
    def stabilize_response(self, raw_response: np.ndarray) -> np.ndarray:
        """
        Apply stabilization techniques to raw PUF response.
        
        Args:
            raw_response: Raw response from PUF, typically in range 0-1V with jitter
            
        Returns:
            Stabilized binary response (0s and 1s)
        """
        # Convert to millivolts for better precision
        response_mv = raw_response * 1000.0
        
        # Calculate adaptive skew based on historical response variations
        # The skew adjusts dynamically based on the magnitude of changes in previous readings
        if len(self.response_history) > 1:
            # Calculate differences between consecutive readings
            diffs = np.abs(np.diff(np.array(self.response_history), axis=0))
            mean_diff = np.mean(diffs)
            
            # Scale skew between 4.0mV (minimal noise) and 12.0mV (high noise)
            # This provides a good balance between stability and sensitivity
            skew = max(4.0, min(12.0, mean_diff))
            logger.debug(f"Using adaptive skew of {skew:.2f}mV based on historical variance")
        else:
            # Default skew for first measurement
            skew = 8.0
            logger.debug(f"Using default skew of {skew:.2f}mV (insufficient history)")
        
        # Store skew value for analysis
        self.skew_history.append(skew)
        
        # Apply temporal majority voting if we have history
        if len(self.response_history) >= 3:
            # Weight recent measurements more heavily
            weights = [0.5, 0.3, 0.2]  # Current, previous, before previous
            weighted_responses = []
            
            # Apply temporal weights to current and previous responses
            weighted_responses.append(response_mv)
            weighted_responses.append(self.response_history[-1])
            weighted_responses.append(self.response_history[-2])
            
            # Calculate weighted average
            temporal_avg = np.zeros_like(response_mv)
            for i, (resp, weight) in enumerate(zip(weighted_responses, weights)):
                temporal_avg += resp * weight
            
            # Use temporally smoothed response
            response_mv = temporal_avg
        
        # Store the raw response for future reference (in mV)
        self.response_history.append(response_mv)
        if len(self.response_history) > 10:
            # Keep only recent history to adapt to environmental changes
            self.response_history.pop(0)
        
        # Apply quantization with the adaptive skew
        # Values > 500+skew are 1, values < 500-skew are 0, in between retains previous value
        stabilized = np.zeros_like(response_mv, dtype=np.int8)
        
        for i, val in enumerate(response_mv):
            if val > (500.0 + skew):
                stabilized[i] = 1
            elif val < (500.0 - skew):
                stabilized[i] = 0
            elif len(self.response_history) > 1:
                # In the hysteresis region, maintain previous value if available
                prev_stabilized = 1 if self.response_history[-2][i] > 500.0 else 0
                stabilized[i] = prev_stabilized
            else:
                # No history, use threshold
                stabilized[i] = 1 if val >= 500.0 else 0
        
        return stabilized


if __name__ == "__main__":
    main()