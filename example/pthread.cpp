#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL double

#include "timer.h"

// uncomment this line to enable the alternative back substitution method
/*#define USE_COLUMN_BACKSUB*/

// use 64-bit IEEE arithmetic (change to "float" to use 32-bit arithmetic)
#define REAL double

// linear system: Ax = b    (A is n x n matrix; b and x are n x 1 vectors)
int n;
REAL *A;
REAL *x;
REAL *b;

// enable/disable debugging output (don't enable for large matrix sizes!)
bool debug_mode = false;

// enable/disable triangular mode (to skip the Gaussian elimination phase)
bool triangular_mode = false;

int numThreads;

typedef struct {
    int startRow;
    int endRow;
    int pivot;
} ThreadData;

pthread_mutex_t mutex_sum;
REAL *partial_sums;

typedef struct {
    int startCol;
    int endCol;
    int row;
} BackSubData;

/*
 * Prints a matrix to standard output in a fixed-width format.
 */
void print_matrix(REAL *mat, int rows, int cols)
{
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            printf("%8.1e ", mat[row*cols + col]);
        }
        printf("\n");
    }
}

void *rand_system_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int startRow = data->startRow;
    int endRow = data->endRow;
    unsigned long seed = data->startRow; 

    for (int row = startRow; row < endRow; row++) {
        int colStart = triangular_mode ? row : 0;
        for (int col = colStart; col < n; col++) {
            if (row != col) {
                seed = (1103515245 * seed + 12345) % (1 << 31);
                A[row * n + col] = (REAL)seed / (REAL)ULONG_MAX;
            } else {
                A[row * n + col] = n / 10.0;
            }
        }
        b[row] = 0.0;
        for (int col = 0; col < n; col++) {
            b[row] += A[row * n + col] * 1.0;
        }
    }

    pthread_exit(NULL);
}

void rand_system_parallel() {
    pthread_t *threads = new pthread_t[numThreads];
    ThreadData *data = new ThreadData[numThreads];
    int chunkSize = (n + numThreads - 1) / numThreads;

    for (int t = 0; t < numThreads; t++) {
        data[t].startRow = t * chunkSize;
        data[t].endRow = std::fmin((t + 1) * chunkSize, n);
        pthread_create(&threads[t], NULL, rand_system_thread, (void *)&data[t]);
    }

    for (int t = 0; t < numThreads; t++) {
        pthread_join(threads[t], NULL);
    }

    delete[] threads;
    delete[] data;
}


/*
 * Reads a linear system of equations from a file in the form of an augmented
 * matrix [A][b].
 */
void read_system(const char *fn)
{
    // open file and read matrix dimensions
    FILE* fin = fopen(fn, "r");
    if (fin == NULL) {
        printf("Unable to open file \"%s\"\n", fn);
        exit(EXIT_FAILURE);
    }
    if (fscanf(fin, "%d\n", &n) != 1) {
        printf("Invalid matrix file format\n");
        exit(EXIT_FAILURE);
    }

    // allocate space for matrices
    A = (REAL*)malloc(sizeof(REAL) * n*n);
    b = (REAL*)malloc(sizeof(REAL) * n);
    x = (REAL*)malloc(sizeof(REAL) * n);

    // verify that memory allocation succeeded
    if (A == NULL || b == NULL || x == NULL) {
        printf("Unable to allocate memory for linear system\n");
        exit(EXIT_FAILURE);
    }

    // read all values
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (fscanf(fin, "%lf", &A[row*n + col]) != 1) {
                printf("Invalid matrix file format\n");
                exit(EXIT_FAILURE);
            }
        }
        if (fscanf(fin, "%lf", &b[row]) != 1) {
            printf("Invalid matrix file format\n");
            exit(EXIT_FAILURE);
        }
        x[row] = 0.0;  
    }
    fclose(fin);
}

void *gaussian_elimination_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int startRow = data->startRow;
    int endRow = data->endRow;
    int pivot = data->pivot;

    for (int row = startRow; row < endRow; row++) {
        REAL coeff = A[row * n + pivot] / A[pivot * n + pivot];
        for (int col = pivot; col < n; col++) {
            A[row * n + col] -= coeff * A[pivot * n + col];
        }
        b[row] -= coeff * b[pivot];
    }
    pthread_exit(NULL);
}

void gaussian_elimination() {
    pthread_t *threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    ThreadData *data = (ThreadData *)malloc(numThreads * sizeof(ThreadData));

    for (int pivot = 0; pivot < n; pivot++) {
        int rowsPerThread = (n - pivot - 1) / numThreads;
        int extra = (n - pivot - 1) % numThreads;

        int currentStartRow = pivot + 1;
        for (int t = 0; t < numThreads; t++) {
            int rowsToHandle = rowsPerThread + (t < extra ? 1 : 0);
            data[t].startRow = currentStartRow;
            data[t].endRow = currentStartRow + rowsToHandle;
            data[t].pivot = pivot;

            if (pthread_create(&threads[t], NULL, gaussian_elimination_thread, (void *)&data[t])) {
                fprintf(stderr, "Error creating thread\n");
                exit(EXIT_FAILURE);
            }
            currentStartRow += rowsToHandle;
        }

        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], NULL);
        }
    }
    free(threads);
    free(data);
}

void *back_substitution_thread(void *arg) {
    BackSubData *data = (BackSubData *)arg;
    REAL sum = 0.0;
    for (int col = data->startCol; col < data->endCol; col++) {
        sum += A[data->row * n + col] * x[col];
    }

    pthread_mutex_lock(&mutex_sum);
    partial_sums[data->row] += sum;
    pthread_mutex_unlock(&mutex_sum);

    pthread_exit(NULL);
}

void back_substitution_row() {
    pthread_t *threads = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    BackSubData *thread_data = (BackSubData *)malloc(numThreads * sizeof(BackSubData));
    partial_sums = (REAL *)calloc(n, sizeof(REAL));

    pthread_mutex_init(&mutex_sum, NULL);

    for (int row = n-1; row >= 0; row--) {
        int remainingCols = n - row - 1;
        int colsPerThread = remainingCols / numThreads;
        int extra = remainingCols % numThreads;

        int startCol = row + 1;
        for (int t = 0; t < numThreads; t++) {
            thread_data[t].row = row;
            thread_data[t].startCol = startCol;
            thread_data[t].endCol = startCol + colsPerThread + (t < extra ? 1 : 0);
            pthread_create(&threads[t], NULL, back_substitution_thread, &thread_data[t]);
            startCol = thread_data[t].endCol;
        }

        for (int t = 0; t < numThreads; t++) {
            pthread_join(threads[t], NULL);
        }

        // Calculate x[row] after all threads are done updating partial sums
        x[row] = (b[row] - partial_sums[row]) / A[row * n + row];
        partial_sums[row] = 0.0;
    }

    free(threads);
    free(thread_data);
    free(partial_sums);
    pthread_mutex_destroy(&mutex_sum);
}


/*
 * Performs backwards substitution on the linear system.
 * (column-oriented version)
 */
void back_substitution_column()
{
    for (int row = 0; row < n; row++) {
        x[row] = b[row];
    }
    for (int col = n-1; col >= 0; col--) {
        x[col] /= A[col*n + col];
        for (int row = 0; row < col; row++) {
            x[row] += -A[row*n + col] * x[col];
        }
    }
}

/*
 * Find the maximum error in the solution (only works for randomly-generated
 * matrices).
 */
REAL find_max_error()
{
    REAL error = 0.0, tmp;
    for (int row = 0; row < n; row++) {
        tmp = fabs(x[row] - 1.0);
        if (tmp > error) {
            error = tmp;
        }
    }
    return error;
}

int main(int argc, char *argv[])
{
    numThreads = 4;

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
            fprintf(stderr, "Usage: %s [-dt] <file|size> [numThreads]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (argc - optind == 2) { // If there are exactly two non-option arguments
        char *endptr;
        long val = strtol(argv[argc - 1], &endptr, 10);

        // Check for various possible errors
        if (val == LONG_MAX || val == LONG_MIN) {
            perror("strtol");
            exit(EXIT_FAILURE);
        }

        if (endptr == argv[argc - 1]) {
            fprintf(stderr, "No digits were found\n");
            exit(EXIT_FAILURE);
        }

        if (*endptr != '\0') {
            fprintf(stderr, "Further characters after number: %s\n", endptr);
            exit(EXIT_FAILURE);
        }

        numThreads = (int)val;
        if (numThreads <= 0) {
            fprintf(stderr, "Invalid number of threads: %ld\n", val);
            exit(EXIT_FAILURE);
        }
    } else if (argc - optind != 1) {
        fprintf(stderr, "Usage: %s [-dt] <file|size> [numThreads]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    long int size = strtol(argv[optind], NULL, 10);
    START_TIMER(init)
    if (size == 0) {
        read_system(argv[optind]);
    } else {
        n = (int)size;
        // Allocate memory for A, b, and x
        A = (REAL*)calloc(n * n, sizeof(REAL));
        b = (REAL*)calloc(n, sizeof(REAL));
        x = (REAL*)calloc(n, sizeof(REAL));
        // Check for memory allocation success
        if (A == NULL || b == NULL || x == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        rand_system_parallel();  // Initialize the system in parallel
    }

    STOP_TIMER(init)

    if (debug_mode) {
        printf("Original A = \n");
        print_matrix(A, n, n);
        printf("Original b = \n");
        print_matrix(b, n, 1);
    }

    // perform gaussian elimination
    START_TIMER(gaus)
    if (!triangular_mode) {
        gaussian_elimination();
    }
    STOP_TIMER(gaus)

    // perform backwards substitution
    START_TIMER(bsub)
#   ifndef USE_COLUMN_BACKSUB
    back_substitution_row();
#   else
    back_substitution_column();
#   endif
    STOP_TIMER(bsub)

    if (debug_mode) {
        printf("Triangular A = \n");
        print_matrix(A, n, n);
        printf("Updated b = \n");
        print_matrix(b, n, 1);
        printf("Solution x = \n");
        print_matrix(x, n, 1);
    }

    // print results
    printf("Nthreads=%2d  ERR=%8.1e  INIT: %8.4fs  GAUS: %8.4fs  BSUB: %8.4fs\n",
            1, find_max_error(),
            GET_TIMER(init), GET_TIMER(gaus), GET_TIMER(bsub));

    // clean up and exit
    free(A);
    free(b);
    free(x);
    return EXIT_SUCCESS;
}