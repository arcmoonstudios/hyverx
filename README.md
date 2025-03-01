# XyVERX

[![XyVERX CI/CD Pipeline](https://github.com/arcmoonstudios/hyverx/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/arcmoonstudios/hyverx/actions/workflows/rust-ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

**Advanced Multi-Dimensional Error Correction System with Hardware Acceleration**

XyVERX is a high-performance error correction library that leverages modern hardware acceleration technologies including CPU vector instructions (AVX2), OpenMP, CUDA, OpenCL, and Intel SYCL to provide exceptional performance across a wide range of devices.

## Features

- **Multi-Dimensional Error Correction**: Implements advanced error correction algorithms across multiple dimensions
- **Hardware Acceleration**: Utilizes various acceleration technologies for optimal performance:
  - AVX2 SIMD instructions for CPU acceleration
  - OpenMP for parallel processing on CPUs
  - CUDA for NVIDIA GPU acceleration
  - OpenCL for cross-platform GPU acceleration
  - Intel SYCL for heterogeneous computing
- **Galois Field Arithmetic**: High-performance finite field arithmetic implementation
- **Reed-Solomon Codes**: Efficient implementation of Reed-Solomon error correction
- **Neural-Assisted Correction**: Novel approaches combining traditional ECC with neural networks
- **Cross-Platform Support**: Works on Linux, Windows, and macOS

## Installation

### Prerequisites

- Rust (nightly toolchain recommended)
- C++ compiler with C++17 support
- For GPU acceleration:
  - CUDA Toolkit 12.x (for NVIDIA GPUs)
  - OpenCL runtime and development libraries
  - Intel oneAPI (for SYCL support)

### From Cargo

```bash
cargo add hyverx
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/arcmoonstudios/hyverx.git
cd hyverx

# Build with default features (CPU only)
cargo build --release

# Build with specific acceleration features
cargo build --release --features cpu,avx2,openmp,cuda,opencl,sycl
```

## Usage

```rust
use hyverx::xypher_grid::XypherGrid;

fn main() {
    // Create a new XypherGrid with default parameters
    let mut grid = XypherGrid::new(256, 16, 8);
    
    // Encode data
    let data = vec![1, 2, 3, 4, 5];
    let encoded = grid.encode(&data);
    
    // Simulate errors (corrupting data)
    let mut corrupted = encoded.clone();
    corrupted[2] = 0;  // Corrupt a byte
    
    // Decode and correct errors
    let decoded = grid.decode(&corrupted);
    
    // Verify the correction
    assert_eq!(data, decoded);
}
```

## Hardware Acceleration

XyVERX automatically selects the best available hardware acceleration method based on the system capabilities and enabled features. You can explicitly control this through feature flags:

```bash
# CPU only with AVX2
cargo build --release --features cpu,avx2

# With CUDA support
cargo build --release --features cpu,avx2,cuda

# With OpenCL support
cargo build --release --features cpu,avx2,opencl

# With all acceleration methods
cargo build --release --features cpu,avx2,openmp,cuda,opencl,sycl
```

## Performance

XyVERX is designed for high performance across various hardware configurations:

- On systems with GPUs, the CUDA or OpenCL backends can provide 10-100x speedup compared to CPU-only implementations
- On systems without GPUs, the AVX2 and OpenMP optimizations still provide significant speedup
- The library automatically falls back to the most efficient available implementation

## Documentation

For detailed documentation, examples, and API reference, visit:

[API Documentation](https://docs.rs/hyverx) (coming soon)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure your code passes all tests and lint checks before submitting a PR.

## License

This project is licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Contact

Lord Xyn - LordXyn@proton.me

Project Link: [https://github.com/arcmoonstudios/hyverx](https://github.com/arcmoonstudios/hyverx) 