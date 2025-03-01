#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
    std::cout << "SYCL Test Program" << std::endl;
    
    // Get information about platforms and devices
    try {
        auto platforms = sycl::platform::get_platforms();
        
        std::cout << "Number of platforms: " << platforms.size() << std::endl;
        
        for (const auto& platform : platforms) {
            std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
            
            auto devices = platform.get_devices();
            std::cout << "  Number of devices: " << devices.size() << std::endl;
            
            for (const auto& device : devices) {
                std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;
                std::cout << "    Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
                std::cout << "    Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
            }
        }
        
        // Create a simple SYCL queue
        sycl::queue queue;
        std::cout << "Default device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        // Run a simple SYCL kernel with proper buffer handling
        constexpr int N = 10;
        std::vector<int> data(N, 0);
        
        {
            // Create a buffer
            sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>(N));
            
            // Submit a command group
            queue.submit([&](sycl::handler& h) {
                // Create an accessor
                auto accessor = buffer.get_access<sycl::access::mode::write>(h);
                
                // Execute the kernel
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                    accessor[i] = i.get(0) * 2;
                });
            });
            
            // Buffer goes out of scope here, data is copied back to host
        }
        
        // Print the results
        std::cout << "Kernel results: ";
        for (int i = 0; i < N; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 