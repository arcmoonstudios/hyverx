#include <stdio.h>
#include <omp.h>

int main() {
    printf("OpenMP Test Program\n");
    printf("Number of available processors: %d\n", omp_get_num_procs());
    printf("Max threads: %d\n", omp_get_max_threads());
    
    #pragma omp parallel
    {
        printf("Hello from thread %d of %d\n", 
               omp_get_thread_num(), omp_get_num_threads());
    }
    
    return 0;
} 