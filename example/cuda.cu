#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define REAL double

// CUDA error check
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

REAL find_max_error(REAL* x, int n) {
    REAL max_error = 0.0;
    for (int i = 0; i < n; i++) {
        REAL error = fabs(x[i] - 1.0);  // Assumes the expected solution is all ones
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
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

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void gaussian_elimination_kernel(REAL *A, REAL *b, int n, int pivot) {
    int row = threadIdx.x + pivot + 1;

    if (row < n) {
        REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
        for (int col = pivot; col < n; col++) {
            A[row * n + col] -= coeff * A[pivot * n + col];
        }
        b[row] -= coeff * b[pivot];
    }
}


// Kernel for backward substitution
__global__ void back_substitution_kernel(REAL *A, REAL *b, REAL *x, int n) {
    int row = n - blockIdx.x - 1;

    REAL sum = 0.0;
    for (int col = row + 1; col < n; col++) {
        sum += A[row * n + col] * x[col];
    }
    x[row] = (b[row] - sum) / A[row * n + row];
}

// Timer macro definitions using CUDA events
float cuda_timer_start, cuda_timer_stop;
cudaEvent_t start, stop;

#define START_TIMER() { \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
}

#define STOP_TIMER() ({ \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float milliseconds = 0; \
    cudaEventElapsedTime(&milliseconds, start, stop); \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop); \
    milliseconds / 1000.0; \
})

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
    long int size = strtol(input, NULL, 10);
    START_TIMER();
    if (size == 0) {
        read_system(input, &A, &b, &x, &n);
    } else {
        n = (int)size;
        generate_random_system(&A, &b, &x, n);
    }
    float init_time = STOP_TIMER();

    // Allocate device memory
    cudaMalloc(&d_A, n * n * sizeof(REAL));
    cudaMalloc(&d_b, n * sizeof(REAL));
    cudaMalloc(&d_x, n * sizeof(REAL));
    cudaCheckError();

    // Copy data from host to device
    cudaMemcpy(d_A, A, n * n * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaCheckError();

    if (debug_mode) {
        printf("Original A = \n");
        print_matrix(A, n, n);
        printf("Original b = \n");
        print_matrix(b, n, 1);
    }

    // Perform Gaussian elimination
    START_TIMER();
    if (!triangular_mode) {
        for (int pivot = 0; pivot < n - 1; pivot++) {
            int threadsPerBlock = n - pivot - 1; // Number of rows below the pivot
            gaussian_elimination_kernel<<<1, threadsPerBlock>>>(d_A, d_b, n, pivot);
            cudaDeviceSynchronize(); // Ensure completion before moving to the next pivot
            cudaCheckError();

            // Optionally: Copy back A and b to host to check the intermediate state
            cudaMemcpy(A, d_A, n * n * sizeof(REAL), cudaMemcpyDeviceToHost);
            cudaMemcpy(b, d_b, n * sizeof(REAL), cudaMemcpyDeviceToHost);
        }
    }
    float gaus_time = STOP_TIMER();


    // Synchronize
    cudaDeviceSynchronize();

    // Perform back substitution
    START_TIMER();
    for (int i = n - 1; i >= 0; i--) {
        REAL sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i * n + i];
    }
    float bsub_time = STOP_TIMER();

    if (debug_mode) {
        printf("Triangular A = \n");
        print_matrix(A, n, n);
        printf("Updated b = \n");
        print_matrix(b, n, 1);
        printf("Solution x = \n");
        print_matrix(x, n, 1);
    }

    // Compute the maximum error
    REAL max_error = find_max_error(x, n);

    // Print results
    printf("Nthreads=%2d  ERR=%8.1e  INIT: %8.4fs  GAUS: %8.4fs  BSUB: %8.4fs\n",
       1, max_error, init_time, gaus_time, bsub_time);

    // Clean up and exit
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    delete[] A;
    delete[] b;
    delete[] x;

    return EXIT_SUCCESS;
}
