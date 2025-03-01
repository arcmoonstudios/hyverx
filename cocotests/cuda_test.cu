#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello from CUDA thread %d\n", threadIdx.x);
}

int main() {
    printf("CUDA Test Program\n");
    cuda_hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
} 