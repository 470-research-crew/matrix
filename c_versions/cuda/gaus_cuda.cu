#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define REAL double

__global__ void gaussianEliminationKernel(REAL* A, REAL* b, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int pivot = blockIdx.y;

    if (row > pivot && row < n) {
        REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
        for (int col = pivot; col < n; col++) {
            A[row * n + col] -= A[pivot * n + col] * coeff;
        }
        b[row] -= b[pivot] * coeff;
    }
}

__global__ void initializeSystemKernel(REAL* A, REAL* b, int n, int triangular_mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;
        int col = idx % n;

        if (!triangular_mode || col >= row) {
            if (row == col) {
                A[idx] = n / 10.0;
            } else {
                A[idx] = (row + col) % 5 + 1;
            }
        } else {
            A[idx] = 0.0;
        }
    }

    if (idx < n) {
        b[idx] = n;
    }
}

__global__ void backSubstitutionKernel(REAL* A, REAL* b, REAL* x, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        REAL sum = b[row];
        for (int col = n - 1; col > row; col--) {
            sum -= A[row * n + col] * x[col];
        }
        x[row] = sum / A[row * n + row];
    }
}

__device__ void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}


__global__ void calculateError(REAL* x, REAL* error, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        REAL localError = fabs(x[idx] - 1.0); 
        atomicMaxDouble(error, localError);
    }
}


void gaussian_elimination(REAL* d_A, REAL* d_b, int n) {
    dim3 blocks((n + 15) / 16, n);
    dim3 threads(16, 1);
    gaussianEliminationKernel<<<blocks, threads>>>(d_A, d_b, n);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *pEnd;
    long int li = strtol(argv[1], &pEnd, 10);
    int n = (int)li;
    if (*pEnd != '\0') {
        fprintf(stderr, "Invalid number: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    size_t bytes = n * n * sizeof(REAL);
    REAL* h_A = (REAL*)malloc(bytes);
    REAL* h_b = (REAL*)malloc(n * sizeof(REAL));
    REAL* h_x = (REAL*)malloc(n * sizeof(REAL));

    REAL* d_A; cudaMalloc(&d_A, bytes);
    REAL* d_b; cudaMalloc(&d_b, n * sizeof(REAL));
    REAL* d_x; cudaMalloc(&d_x, n * sizeof(REAL));

    // Initialize matrices A and b
    dim3 initBlocks((n * n + 1023) / 1024);
    dim3 initThreads(1024);
    initializeSystemKernel<<<initBlocks, initThreads>>>(d_A, d_b, n, 0);
    cudaDeviceSynchronize();

    // Perform Gaussian elimination
    gaussian_elimination(d_A, d_b, n); // Ensure this function is updated to include the back substitution

    // Copy back the result to host
    cudaMemcpy(h_x, d_x, n * sizeof(REAL), cudaMemcpyDeviceToHost);

    // Calculate error on the device
    REAL* d_error; cudaMalloc(&d_error, sizeof(REAL));
    cudaMemset(d_error, 0, sizeof(REAL));
    calculateError<<<initBlocks, initThreads>>>(d_x, d_error, n);
    cudaDeviceSynchronize();

    // Copy error back to host and print
    REAL h_error;
    cudaMemcpy(&h_error, d_error, sizeof(REAL), cudaMemcpyDeviceToHost);
    printf("Max error: %f\n", h_error);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_error);
    free(h_A);
    free(h_b);
    free(h_x);

    return 0;
}
