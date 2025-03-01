// build.rs
use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Get environment variables
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // Detect CPU architecture features
    let has_avx2 = is_avx2_supported();

    println!("cargo:rerun-if-changed=src/hardware/cpu/avx2_kernels.cpp");
    println!("cargo:rerun-if-changed=src/hardware/cpu/openmp_kernels.cpp");
    println!("cargo:rerun-if-changed=src/hardware/include/avx2_kernels.hpp");
    println!("cargo:rerun-if-changed=src/hardware/include/openmp_kernels.hpp");

    // Check if CUDA, SYCL, and OpenCL are available - using exact paths for this system
    let cuda_path =
        Some(Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6").to_path_buf());
    let sycl_path = Some(
        Path::new("C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include").to_path_buf(),
    );
    let opencl_path = Some(
        Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64").to_path_buf(),
    );

    // Print includes path information for diagnostic purposes
    if let Some(ref path) = sycl_path {
        println!("cargo:rustc-env=SYCL_PATH={}", path.display());
        println!(
            "cargo:warning=Found SYCL installation at: {}",
            path.display()
        );

        // Add SYCL include path to compiler flags
        println!("cargo:rustc-env=CPLUS_INCLUDE_PATH={}", path.display());
        println!("cargo:rustc-cfg=feature=\"sycl\"");
        println!("cargo:rustc-cfg=SYCL_ENABLED");
    }

    if let Some(ref path) = cuda_path {
        println!("cargo:rustc-env=CUDA_PATH={}", path.display());
        println!(
            "cargo:warning=Found CUDA installation at: {}",
            path.display()
        );
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }

    if let Some(ref path) = opencl_path {
        println!("cargo:rustc-env=OPENCL_PATH={}", path.display());
        println!(
            "cargo:warning=Found OpenCL installation at: {}",
            path.display()
        );
        println!("cargo:rustc-cfg=feature=\"opencl\"");
    }

    // Compile AVX2 kernels
    compile_avx2_kernels(has_avx2, &target_os);

    // Compile OpenMP kernels
    compile_openmp_kernels(&target_os);

    // Link the compiled libraries
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=avx2_kernels");
    println!("cargo:rustc-link-lib=static=openmp_kernels");

    // Link system libraries
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        if is_openmp_supported() {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    } else if target_os == "windows" {
        // For MSVC, we need to link against the Microsoft C++ runtime libraries
        // Don't use stdc++ on Windows, as it's not available with MSVC
        if cfg!(target_env = "msvc") {
            // Link with MSVC C++ runtime instead
            println!("cargo:rustc-link-lib=dylib=msvcrt");
            println!("cargo:rustc-link-lib=dylib=msvcprt");
        } else {
            // MinGW uses stdc++
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }

        if is_openmp_supported() {
            // On Windows with Intel oneAPI, we need the Intel OpenMP library
            println!(
                "cargo:rustc-link-search=C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/lib"
            );
            println!("cargo:rustc-link-lib=dylib=libiomp5md");
        }
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
        if is_openmp_supported() {
            println!("cargo:rustc-link-lib=dylib=omp");
        }
    }
}

fn compile_avx2_kernels(has_avx2: bool, target_os: &str) {
    let mut config = cc::Build::new();

    // Configure compiler
    config
        .cpp(true)
        .file("src/hardware/cpu/avx2_kernels.cpp")
        .include("src/hardware/include")
        .opt_level(3);

    // Add platform-specific settings
    if has_avx2 {
        config.flag("-mavx2");
        println!("cargo:rustc-cfg=avx2");
    }

    // Add platform-specific settings
    if target_os == "linux" || target_os == "macos" {
        config.flag("-std=c++17");
    } else if target_os == "windows" {
        config.flag("/std:c++17");
    }

    // Compile the library
    config.compile("avx2_kernels");
}

fn compile_openmp_kernels(target_os: &str) {
    let mut config = cc::Build::new();

    // Intel OneAPI path for OpenMP
    let intel_include_path = "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include";

    // Configure compiler
    config
        .cpp(true)
        .file("src/hardware/cpu/openmp_kernels.cpp")
        .include("src/hardware/include")
        .include(intel_include_path)
        .opt_level(3);

    // Include SYCL headers when available
    if let Ok(sycl_path) = env::var("SYCL_PATH") {
        config.include(&sycl_path);
        config.define("SYCL_ENABLED", None);
    }

    // Include CUDA headers when available
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        config.include(format!("{}/include", cuda_path));
        config.define("CUDA_ENABLED", None);
    }

    // Add OpenMP support if available
    if is_openmp_supported() {
        if target_os == "linux" || target_os == "macos" {
            config.flag("-fopenmp");
        } else if target_os == "windows" {
            config.flag("/openmp");
        }
        println!("cargo:rustc-cfg=feature=\"openmp\"");
    }

    // Add platform-specific settings
    if target_os == "windows" {
        config.flag("/std:c++17"); // Use C++17 instead of C++23 for better compatibility
    } else if target_os == "linux" || target_os == "macos" {
        config.flag("-std=c++17"); // Use C++17 instead of C++23 for better compatibility
    }

    // Compile the library
    config.compile("openmp_kernels");
}

fn is_avx2_supported() -> bool {
    // Check if AVX2 is supported on the build machine
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        // Try to detect AVX2 support
        let output = Command::new("grep")
            .args(["-q", "avx2", "/proc/cpuinfo"])
            .output();

        if let Ok(output) = output {
            return output.status.success();
        }

        // Fallback detection on non-Linux
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("wmic").args(["cpu", "get", "Name"]).output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.to_lowercase().contains("avx2") {
                    return true;
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl")
                .args(["-n", "machdep.cpu.features"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.to_uppercase().contains("AVX2") {
                    return true;
                }
            }
        }
    }

    false
}

fn is_openmp_supported() -> bool {
    // Try to compile a small OpenMP program
    let temp_dir = env::temp_dir();
    let src_path = temp_dir.join("openmp_check.cpp");
    let out_path = temp_dir.join("openmp_check");

    // Intel OneAPI path for OpenMP
    let intel_include_path = "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include";

    // Create a simple OpenMP test program
    std::fs::write(
        &src_path,
        r#"
        #include <omp.h>
        int main() {
            int x = 0;
            #pragma omp parallel for
            for (int i = 0; i < 10; i++) {
                x += i;
            }
            return x > 0 ? 0 : 1;
        }
    "#,
    )
    .unwrap_or(());

    // Try to compile with OpenMP
    let status = if cfg!(target_os = "windows") {
        Command::new("cl")
            .args([
                "/openmp",
                format!("/I{}", intel_include_path).as_str(),
                src_path.to_str().unwrap(),
                "/Fe:",
                out_path.to_str().unwrap(),
            ])
            .status()
    } else {
        Command::new("g++")
            .args([
                "-fopenmp",
                "-o",
                out_path.to_str().unwrap(),
                src_path.to_str().unwrap(),
            ])
            .status()
    };

    // Clean up temporary files
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);

    // Return true if compilation succeeded
    status.map(|s| s.success()).unwrap_or(false)
}

// The following functions are no longer used since we're hardcoding paths
// Left as commented code for reference in case dynamic detection is needed in the future
/*
fn detect_sycl_installation(target_os: &str) -> Option<std::path::PathBuf> {
    // Implementation removed for brevity
    None
}

fn detect_cuda_installation() -> Option<std::path::PathBuf> {
    // Implementation removed for brevity
    None
}

fn detect_opencl_installation(target_os: &str) -> Option<std::path::PathBuf> {
    // Implementation removed for brevity
    None
}
*/
