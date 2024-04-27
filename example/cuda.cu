#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <string.h>

#define REAL double

// CUDA error check
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to allocate memory and initialize the matrix and vectors
void generate_random_system(REAL** A, REAL** b, REAL** x, int n) {
    // Seed the random number generator to get different results each time
    srand(time(NULL));

    // Allocate memory for A, b, x
    *A = new REAL[n * n];
    *b = new REAL[n];
    *x = new REAL[n]; // This will be the solution vector; initialized later

    // Fill the matrix A and vector b
    for (int i = 0; i < n; i++) {
        (*b)[i] = 0.0; // Initialize b[i] to zero for accumulation
        for (int j = 0; j < n; j++) {
            if (i == j) {
                (*A)[i * n + j] = n / 10.0; // Set diagonal dominance
            } else {
                (*A)[i * n + j] = (REAL)rand() / RAND_MAX; // Random double between 0.0 and 1.0
            }
            (*b)[i] += (*A)[i * n + j]; // Accumulate to form b[i]
        }
    }

    // Optionally initialize x to some default values (e.g., zeros)
    for (int i = 0; i < n; i++) {
        (*x)[i] = 0.0; // Not necessary for solving, but good for initialization
    }
}

// Function to read matrix A and vector b from a file
void read_system(const char* filename, REAL** A, REAL** b, REAL** x, int* n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read the dimension of the matrix
    if (fscanf(file, "%d", n) != 1) {
        fprintf(stderr, "Invalid matrix file format\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Allocate memory for A, b, x
    *A = new REAL[*n * *n];
    *b = new REAL[*n];
    *x = new REAL[*n]; // This will be the solution vector; initialized later

    // Read the matrix A and vector b
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            if (fscanf(file, "%lf", &(*A)[i * *n + j]) != 1) {
                fprintf(stderr, "Invalid matrix file format while reading A[%d][%d]\n", i, j);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
        if (fscanf(file, "%lf", &(*b)[i]) != 1) {
            fprintf(stderr, "Invalid matrix file format while reading b[%d]\n", i);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    // Optionally initialize x to some default values (e.g., zeros)
    for (int i = 0; i < *n; i++) {
        (*x)[i] = 0.0; // Initialize solution vector to zero
    }

    fclose(file);
}

// Kernel to perform Gaussian elimination
__global__ void gaussian_elimination_kernel(REAL *A, REAL *b, int n, int pivot) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + pivot + 1;
    if (row < n) {
        REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
        for (int col = pivot; col < n; col++) { // Update column from pivot to ensure 0 below pivot
            A[row * n + col] -= A[pivot * n + col] * coeff;
        }
        b[row] -= b[pivot] * coeff;
    }
}

// Kernel for backward substitution
__global__ void back_substitution_kernel(REAL *A, REAL *b, REAL *x, int n) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        REAL tmp = b[row];
        for (int col = row + 1; col < n; col++) {
            tmp -= A[row * n + col] * x[col];
        }
        x[row] = tmp / A[row * n + row];
    }
}

// Timer macro definitions using CUDA events
float cuda_timer_start, cuda_timer_stop;
cudaEvent_t start, stop;

#define START_TIMER() { \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start, 0); \
}

#define STOP_TIMER() { \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&cuda_timer_start, start, stop); \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop); \
}

#define GET_TIMER() (cuda_timer_start / 1000.0) // Return in seconds

// Function to print matrices (host side)
void print_matrix(REAL *mat, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            printf("%8.1e ", mat[row * cols + col]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int n;          // Matrix size
    bool debug_mode = false;
    bool triangular_mode = false;
    
    // Parse command line arguments
    int c;
    while ((c = getopt(argc, argv, "dt")) != -1) {
        switch (c) {
        case 'd':
            debug_mode = true;
            break;
        case 't':
            triangular_mode = true;
            break;
        default:
            printf("Usage: %s [-dt] <file|size>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (optind != argc - 1) {
        printf("Usage: %s [-dt] <file|size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    REAL *A, *b, *x; // Host pointers
    REAL *d_A, *d_b, *d_x; // Device pointers

    // Determine if input is a file or a matrix size
    char* input = argv[optind];
    int size = strtol(input, NULL, 10);
    if (size == 0) {
        // Read system from file
        read_system(input, &A, &b, &x, &n);
    } else {
        n = size;
        // Generate random system of size n
        generate_random_system(&A, &b, &x, n);
    }

    // Allocate device memory
    cudaMalloc(&d_A, n * n * sizeof(REAL));
    cudaMalloc(&d_b, n * sizeof(REAL));
    cudaMalloc(&d_x, n * sizeof(REAL));
    cudaCheckError();

    // Copy data from host to device
    cudaMemcpy(d_A, A, n * n * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaCheckError();

    int threadsPerBlock = 256; // Common number of threads per block

    // Running Gaussian elimination
    if (!triangular_mode) {
        for (int i = 0; i < n - 1; i++) {
            int blocks = (n - i - 1 + threadsPerBlock - 1) / threadsPerBlock;
            gaussian_elimination_kernel<<<blocks, threadsPerBlock>>>(d_A, d_b, n, i);
            cudaCheckError();
        }
    }

    // Running back substitution
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    back_substitution_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_b, d_x, n);
    cudaCheckError();

    // Copy back the result
    cudaMemcpy(x, d_x, n * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Example of how to print and check error if in debug mode
    if (debug_mode) {
        printf("Final solution x = \n");
        print_matrix(x, n, 1);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    delete[] A;
    delete[] b;
    delete[] x;

    return 0;
}
