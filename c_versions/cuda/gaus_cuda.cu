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

void gaussian_elimination(REAL* d_A, REAL* d_b, int n) {
    dim3 blocks((n + 15) / 16, n);
    dim3 threads(16, 1);
    gaussianEliminationKernel<<<blocks, threads>>>(d_A, d_b, n);
    cudaDeviceSynchronize();
}

int main() {
    int n = 10;
    size_t bytes = n * n * sizeof(REAL);
    REAL* h_A = (REAL*)malloc(bytes);
    REAL* h_b = (REAL*)malloc(n * sizeof(REAL));

    REAL* d_A; cudaMalloc(&d_A, bytes);
    REAL* d_b; cudaMalloc(&d_b, n * sizeof(REAL));

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(REAL), cudaMemcpyHostToDevice);

    gaussian_elimination(d_A, d_b, n);

    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, n * sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    free(h_A);
    free(h_b);

    return 0;
}
