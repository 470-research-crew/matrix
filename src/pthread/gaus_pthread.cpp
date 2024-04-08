#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <vector>

#define REAL double

#include "timer.h"

// Uncomment this line to enable the alternative back substitution method
// #define USE_COLUMN_BACKSUB

// Linear system: Ax = b (A is n x n matrix; b and x are n x 1 vectors)
int n;
REAL *A;
REAL *x;
REAL *b;

// Enable/disable debugging output (don't enable for large matrix sizes!)
bool debug_mode = false;

// Enable/disable triangular mode (to skip the Gaussian elimination phase)
bool triangular_mode = false;

int numThreads;

struct ThreadData {
    int startRow;
    int endRow;
    int pivot;
};

void rand_system() {
    A = new REAL[n*n];
    b = new REAL[n];
    x = new REAL[n];

    if (!A || !b || !x) {
        std::cerr << "Unable to allocate memory for linear system\n";
        exit(EXIT_FAILURE);
    }

    unsigned long seed = 0;

    for (int row = 0; row < n; ++row) {
        int col = triangular_mode ? row : 0;
        for (; col < n; ++col) {
            if (row != col) {
                seed = (1103515245 * seed + 12345) % (1UL << 31);
                A[row*n + col] = static_cast<REAL>(seed) / static_cast<REAL>(ULONG_MAX);
            } else {
                A[row*n + col] = n / 10.0;
            }
        }
    }

    for (int row = 0; row < n; ++row) {
        b[row] = 0.0;
        for (int col = 0; col < n; ++col) {
            b[row] += A[row*n + col];
        }
    }
}

void read_system(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        std::cerr << "Unable to open file \"" << filename << "\"\n";
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d\n", &n) != 1) {
        std::cerr << "Invalid matrix file format\n";
        exit(EXIT_FAILURE);
    }

    A = new REAL[n*n];
    b = new REAL[n];
    x = new REAL[n];

    if (!A || !b || !x) {
        std::cerr << "Unable to allocate memory for linear system\n";
        exit(EXIT_FAILURE);
    }

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            if (fscanf(file, "%lf", &A[row*n + col]) != 1) {
                std::cerr << "Invalid matrix file format\n";
                exit(EXIT_FAILURE);
            }
        }
        if (fscanf(file, "%lf", &b[row]) != 1) {
            std::cerr << "Invalid matrix file format\n";
            exit(EXIT_FAILURE);
        }
        x[row] = 0.0;
    }

    fclose(file);
}

void *gaussian_elimination_thread(void *arg) {
    ThreadData *data = static_cast<ThreadData*>(arg);
    int startRow = data->startRow;
    int endRow = data->endRow;
    int pivot = data->pivot;

    for (int row = startRow; row < endRow; ++row) {
        REAL coeff = A[row*n + pivot] / A[pivot*n + pivot];
        A[row*n + pivot] = 0.0;
        for (int col = pivot + 1; col < n; ++col) {
            A[row*n + col] -= A[pivot*n + col] * coeff;
        }
        b[row] -= b[pivot] * coeff;
    }

    pthread_exit(NULL);
}

void gaussian_elimination() {
    pthread_t *threads = new pthread_t[numThreads];
    ThreadData *data = new ThreadData[numThreads];

    for (int pivot = 0; pivot < n; ++pivot) {
        for (int t = 0; t < numThreads; ++t) {
            data[t].startRow = pivot + 1 + (n - pivot - 1) / numThreads * t;
            data[t].endRow = pivot + 1 + (n - pivot - 1) / numThreads * (t + 1);
            data[t].pivot = pivot;

            if (pthread_create(&threads[t], NULL, gaussian_elimination_thread, &data[t])) {
                std::cerr << "Error creating thread\n";
                exit(EXIT_FAILURE);
            }
        }

        for (int t = 0; t < numThreads; ++t) {
            pthread_join(threads[t], NULL);
        }
    }

    delete[] threads;
    delete[] data;
}

void back_substitution_row() {
    REAL tmp;
    for (int row = n-1; row >= 0; --row) {
        tmp = b[row];
        for (int col = row + 1; col < n; ++col) {
            tmp -= A[row*n + col] * x[col];
        }
        x[row] = tmp / A[row*n + row];
    }
}

void back_substitution_column() {
    for (int row = 0; row < n; ++row) {
        x[row] = b[row];
    }
    for (int col = n-1; col >= 0; --col) {
        x[col] /= A[col*n + col];
        for (int row = 0; row < col; ++row) {
            x[row] -= A[row*n + col] * x[col];
        }
    }
}

REAL find_max_error() {
    REAL max_error = 0.0;
    for (int row = 0; row < n; ++row) {
        REAL error = std::fabs(x[row] - 1.0);
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

void print_matrix(REAL *matrix, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            printf("%8.1e ", matrix[row*cols + col]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    numThreads = 4; // Default number of threads

    int option;
    while ((option = getopt(argc, argv, "dt")) != -1) {
        switch (option) {
            case 'd':
                debug_mode = true;
                break;
            case 't':
                triangular_mode = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-dt] <file|size> [numThreads]\n";
                exit(EXIT_FAILURE);
        }
    }

    if (argc - optind == 2) {
        char *endptr;
        long val = std::strtol(argv[argc - 1], &endptr, 10);
        if ((val == LONG_MAX || val == LONG_MIN) && errno == ERANGE) {
            perror("strtol");
            exit(EXIT_FAILURE);
        }
        if (endptr == argv[argc - 1]) {
            std::cerr << "No digits were found\n";
            exit(EXIT_FAILURE);
        }
        if (*endptr != '\0') {
            std::cerr << "Further characters after number: " << endptr << "\n";
            exit(EXIT_FAILURE);
        }
        numThreads = static_cast<int>(val);
        if (numThreads <= 0) {
            std::cerr << "Invalid number of threads: " << val << "\n";
            exit(EXIT_FAILURE);
        }
    } else if (argc - optind != 1) {
        std::cerr << "Usage: " << argv[0] << " [-dt] <file|size> [numThreads]\n";
        exit(EXIT_FAILURE);
    }

    // Initialize linear system
    START_TIMER(init)
    if (isdigit(argv[optind][0])) {
        n = std::atoi(argv[optind]);
        rand_system();
    } else {
        read_system(argv[optind]);
    }
    STOP_TIMER(init)

    // Optionally print the original matrices for debugging
    if (debug_mode) {
        std::cout << "Original A = \n";
        print_matrix(A, n, n);
        std::cout << "Original b = \n";
        print_matrix(b, n, 1);
    }

    // Perform Gaussian elimination
    START_TIMER(gaus)
    if (!triangular_mode) {
        gaussian_elimination();
    }
    STOP_TIMER(gaus)

    // Perform backwards substitution
    START_TIMER(bsub)
#ifndef USE_COLUMN_BACKSUB
    back_substitution_row();
#else
    back_substitution_column();
#endif
    STOP_TIMER(bsub)

    // Optionally print the resulting matrices for debugging
    if (debug_mode) {
        std::cout << "Triangular A = \n";
        print_matrix(A, n, n);
        std::cout << "Updated b = \n";
        print_matrix(b, n, 1);
        std::cout << "Solution x = \n";
        print_matrix(x, n, 1);
    }

    // Print results
    std::cout << "Nthreads=" << numThreads << "  ERR=" << find_max_error()
              << "  INIT: " << GET_TIMER(init) << "s  GAUS: " << GET_TIMER(gaus) 
              << "s  BSUB: " << GET_TIMER(bsub) << "s\n";

    // Cleanup
    delete[] A;
    delete[] b;
    delete[] x;

    return EXIT_SUCCESS;
}

