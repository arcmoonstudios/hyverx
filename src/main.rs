//! HyVERX: Advanced Multi-Dimensional Error Correction System with Hardware Acceleration
//!
//! This application provides a highly optimized implementation of advanced error correction
//! algorithms with hardware acceleration support for CUDA, AVX2, and OpenCL.
//!
//! # Features
//!
//! - Multi-dimensional error detection and correction with parallel processing
//! - Dynamic algorithm allocation based on error characteristics
//! - Hardware acceleration via AVX2/OpenMP SIMT and CUDA/cuDNN tensor cores
//! - Neural-symbolic integration for complex error pattern recognition
//! - Precomputed correction matrices for zero-cost abstraction
//! - Cross-platform support with flexible hardware targeting

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, Command};
use hyverx::config::{Config, HardwareTarget};
use hyverx::prelude::*;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set global default subscriber")?;

    // Parse command line arguments
    let matches = Command::new("hyverx")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Arcmoon Studios")
        .about("Advanced Multi-Dimensional Error Correction System with Hardware Acceleration")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Input file to process")
                .required(true)
                .value_name("FILE"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file")
                .required(true)
                .value_name("FILE"),
        )
        .arg(
            Arg::new("ecc-size")
                .short('e')
                .long("ecc-size")
                .help("Size of error correction code in bytes")
                .default_value("32")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("dimensions")
                .short('d')
                .long("dimensions")
                .help("Number of dimensions for error correction")
                .default_value("16")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("hardware")
                .short('h')
                .long("hardware")
                .help("Hardware acceleration target")
                .value_parser(["auto", "cpu", "cuda", "opencl", "all"])
                .default_value("auto"),
        )
        .arg(
            Arg::new("generate-tables")
                .short('g')
                .long("generate-tables")
                .help("Generate precomputed lookup tables")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("table-path")
                .short('t')
                .long("table-path")
                .help("Path to store precomputed lookup tables")
                .default_value("./tables"),
        )
        .arg(
            Arg::new("threads")
                .short('j')
                .long("threads")
                .help("Number of threads to use (0 for auto)")
                .default_value("0")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    // Extract command line arguments
    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    let ecc_size = *matches.get_one::<usize>("ecc-size").unwrap();
    let dimensions = *matches.get_one::<usize>("dimensions").unwrap();
    let hardware_str = matches.get_one::<String>("hardware").unwrap();
    let generate_tables = matches.get_flag("generate-tables");
    let table_path = matches.get_one::<String>("table-path").unwrap();
    let threads = *matches.get_one::<usize>("threads").unwrap();
    let verbose = matches.get_flag("verbose");

    // Set up hardware target
    let hardware_target = match hardware_str.as_str() {
        "auto" => HardwareTarget::Auto,
        "cpu" => HardwareTarget::Cpu,
        "cuda" => HardwareTarget::Cuda,
        "opencl" => HardwareTarget::OpenCL,
        "all" => HardwareTarget::All,
        _ => HardwareTarget::Auto,
    };

    // Configure the application
    let config = Config::new()
        .with_ecc_size(ecc_size)
        .with_dimensions(dimensions)
        .with_hardware_target(hardware_target)
        .with_table_path(PathBuf::from(table_path))
        .with_threads(threads)
        .with_verbose(verbose);

    // Display configuration
    info!("HyVERX Configuration:");
    info!("  Input: {}", input_path);
    info!("  Output: {}", output_path);
    info!("  ECC Size: {} bytes", ecc_size);
    info!("  Dimensions: {}", dimensions);
    info!("  Hardware Target: {:?}", hardware_target);
    let thread_str = threads.to_string();
    info!(
        "  Threads: {}",
        if threads == 0 { "auto" } else { &thread_str }
    );

    // Initialize the HyVERX system
    info!("Initializing HyVERX system...");
    let mut hyverx = HyVerxSystem::new(config)?;

    // Generate lookup tables if requested
    if generate_tables {
        info!("Generating precomputed lookup tables...");
        let start = Instant::now();
        hyverx.generate_lookup_tables()?;
        info!("Lookup tables generated in {:?}", start.elapsed());
    }

    // Read input data
    info!("Reading input data from {}...", input_path);
    let input_data = std::fs::read(input_path).context("Failed to read input file")?;

    // Process the data
    info!(
        "Processing data with {} dimensions and {} ECC bytes...",
        dimensions, ecc_size
    );
    let start = Instant::now();
    let result = hyverx.process_data(&input_data)?;
    let elapsed = start.elapsed();
    info!("Processing completed in {:?}", elapsed);

    // Calculate performance metrics
    let throughput = (input_data.len() as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    info!(
        "Throughput: {:.2} MiB/s ({:.2} Mbps)",
        throughput,
        throughput * 8.0
    );

    // Get error correction statistics
    let stats = hyverx.get_statistics();
    info!("Error correction statistics: {:#?}", stats);

    // Write output data
    info!("Writing output to {}...", output_path);
    std::fs::write(output_path, result).context("Failed to write output file")?;

    info!("HyVERX processing completed successfully");
    Ok(())
}
