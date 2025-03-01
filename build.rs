// build.rs
use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Get environment variables
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "".to_string());

    // Print target info for diagnostics
    println!("cargo:warning=Building for target OS: {}", target_os);
    println!("cargo:warning=Target environment: {}", target_env);

    // Detect CPU architecture features
    let has_avx2 = is_avx2_supported();
    println!("cargo:warning=AVX2 support detected: {}", has_avx2);

    println!("cargo:rerun-if-changed=src/hardware/cpu/avx2_kernels.cpp");
    println!("cargo:rerun-if-changed=src/hardware/cpu/openmp_kernels.cpp");
    println!("cargo:rerun-if-changed=src/hardware/include/avx2_kernels.hpp");
    println!("cargo:rerun-if-changed=src/hardware/include/openmp_kernels.hpp");

    // Path detection based on platform
    let (cuda_path, sycl_path, opencl_path) = detect_paths(&target_os);

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
    compile_avx2_kernels(has_avx2, &target_os, &target_env);

    // Compile OpenMP kernels
    compile_openmp_kernels(&target_os, &target_env);

    // Link the compiled libraries
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=avx2_kernels");
    println!("cargo:rustc-link-lib=static=openmp_kernels");

    // Link system libraries based on platform
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        if is_openmp_supported() {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    } else if target_os == "windows" {
        // For MSVC, use MSVC C++ runtime libraries
        if target_env == "msvc" {
            println!("cargo:rustc-link-lib=dylib=msvcrt");
            println!("cargo:rustc-link-lib=dylib=msvcprt");
        } else {
            // MinGW uses stdc++
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }

        if is_openmp_supported() {
            // On Windows with Intel oneAPI, use Intel OpenMP library
            if let Ok(oneapi_path) = env::var("ONEAPI_ROOT") {
                println!("cargo:rustc-link-search={}/compiler/latest/lib", oneapi_path);
            } else {
                println!("cargo:rustc-link-search=C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/lib");
            }
            println!("cargo:rustc-link-lib=dylib=libiomp5md");
        }
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
        if is_openmp_supported() {
            println!("cargo:rustc-link-lib=dylib=omp");
        }
    }
}

fn detect_paths(target_os: &str) -> (Option<std::path::PathBuf>, Option<std::path::PathBuf>, Option<std::path::PathBuf>) {
    let cuda_path;
    let sycl_path;
    let opencl_path;

    if target_os == "windows" {
        cuda_path = Some(Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6").to_path_buf());
        sycl_path = Some(Path::new("C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include").to_path_buf());
        opencl_path = Some(Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64").to_path_buf());
    } else if target_os == "linux" {
        // Linux paths
        cuda_path = if Path::new("/usr/local/cuda-12.6").exists() {
            Some(Path::new("/usr/local/cuda-12.6").to_path_buf())
        } else if Path::new("/usr/local/cuda").exists() {
            Some(Path::new("/usr/local/cuda").to_path_buf())
        } else {
            None
        };

        sycl_path = if Path::new("/opt/intel/oneapi/compiler/latest/include").exists() {
            Some(Path::new("/opt/intel/oneapi/compiler/latest/include").to_path_buf())
        } else {
            None
        };

        opencl_path = if Path::new("/usr/include/CL").exists() {
            Some(Path::new("/usr/include/CL").to_path_buf())
        } else {
            None
        };
    } else {
        // Default to None for other platforms
        cuda_path = None;
        sycl_path = None;
        opencl_path = None;
    }

    (cuda_path, sycl_path, opencl_path)
}

fn compile_avx2_kernels(has_avx2: bool, target_os: &str, target_env: &str) {
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

    // Add platform-specific settings based on OS and environment
    if target_os == "linux" || target_os == "macos" {
        config.flag("-std=c++17");
    } else if target_os == "windows" {
        if target_env == "msvc" {
            config.flag("/std:c++17");
        } else {
            // For MinGW on Windows
            config.flag("-std=c++17");
        }
    }

    // Compile the library
    config.compile("avx2_kernels");
}

fn compile_openmp_kernels(target_os: &str, target_env: &str) {
    let mut config = cc::Build::new();

    // Intel OneAPI path - platform specific
    let intel_include_path = if target_os == "windows" {
        "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include"
    } else if target_os == "linux" {
        "/opt/intel/oneapi/compiler/latest/include"
    } else if target_os == "macos" {
        "/opt/intel/oneapi/compiler/latest/include"
    } else {
        ""
    };

    // Configure compiler
    config
        .cpp(true)
        .file("src/hardware/cpu/openmp_kernels.cpp")
        .include("src/hardware/include");

    if !intel_include_path.is_empty() {
        config.include(intel_include_path);
    }

    config.opt_level(3);

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
            if target_env == "msvc" {
                config.flag("/openmp");
            } else {
                // For MinGW on Windows
                config.flag("-fopenmp");
            }
        }
        println!("cargo:rustc-cfg=feature=\"openmp\"");
    }

    // Add platform-specific settings
    if target_os == "windows" {
        if target_env == "msvc" {
            config.flag("/std:c++17"); // Use C++17 instead of C++23 for better compatibility
        } else {
            // For MinGW on Windows
            config.flag("-std=c++17");
        }
    } else if target_os == "linux" || target_os == "macos" {
        config.flag("-std=c++17"); // Use C++17 instead of C++23 for better compatibility
    }

    // Compile the library
    config.compile("openmp_kernels");
}

fn is_avx2_supported() -> bool {
    // Check if AVX2 is supported on the build machine
    if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        // Linux detection
        if cfg!(target_os = "linux") {
            let output = Command::new("grep")
                .args(["-q", "avx2", "/proc/cpuinfo"])
                .output();

            if let Ok(output) = output {
                return output.status.success();
            }
        }

        // Windows detection
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("wmic").args(["cpu", "get", "Name"]).output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.to_lowercase().contains("avx2") {
                    return true;
                }
            }
            
            // Alternative detection for Windows
            if let Ok(output) = Command::new("powershell")
                .args(["-Command", "(Get-WmiObject Win32_Processor).Name"])
                .output() 
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.to_lowercase().contains("avx2") {
                    return true;
                }
            }
        }

        // macOS detection
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

    // Fallback: check RUSTFLAGS for target features
    if let Ok(rustflags) = env::var("RUSTFLAGS") {
        if rustflags.contains("target-feature=+avx2") {
            return true;
        }
    }

    false
}

fn is_openmp_supported() -> bool {
    // Try to compile a small OpenMP program
    let temp_dir = env::temp_dir();
    let src_path = temp_dir.join("openmp_check.cpp");
    let out_path = temp_dir.join("openmp_check");

    // Get target information
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "unknown".to_string());
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "".to_string());

    // Intel OneAPI path - platform specific
    let intel_include_path = if target_os == "windows" {
        "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/include"
    } else if target_os == "linux" {
        "/opt/intel/oneapi/compiler/latest/include"
    } else {
        ""
    };

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
    let status = if target_os == "windows" && target_env == "msvc" {
        Command::new("cl")
            .args([
                "/openmp",
                &format!("/I{}", intel_include_path),
                src_path.to_str().unwrap(),
                "/Fe:",
                out_path.to_str().unwrap(),
            ])
            .status()
    } else if target_os == "windows" {
        // MinGW on Windows
        Command::new("g++")
            .args([
                "-fopenmp",
                "-o",
                out_path.to_str().unwrap(),
                src_path.to_str().unwrap(),
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
    match status {
        Ok(s) => {
            let success = s.success();
            println!("cargo:warning=OpenMP support detected: {}", success);
            success
        },
        Err(e) => {
            println!("cargo:warning=OpenMP compilation test failed: {}", e);
            false
        }
    }
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
