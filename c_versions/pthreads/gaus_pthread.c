#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define REAL double
#define NUM_THREADS 4 // Adjust based on your system capabilities

typedef struct ThreadData {
    int thread_id;
    REAL* A;
    REAL* b;
    REAL* x;
    int n;
    pthread_barrier_t *barrier; // For synchronization between threads
} ThreadData;

// Function prototypes
void* gaussianEliminationThread(void* threadarg);
void* backSubstitutionThread(void* threadarg);
void initializeSystem(REAL* A, REAL* b, int n);

void* gaussianEliminationThread(void* threadarg) {
    ThreadData* data = (ThreadData*) threadarg;
    int tid = data->thread_id;
    REAL* A = data->A;
    REAL* b = data->b;
    int n = data->n;

    for (int pivot = 0; pivot < n - 1; pivot++) {
        pthread_barrier_wait(data->barrier);
        for (int row = pivot + 1 + tid; row < n; row += NUM_THREADS) {
            REAL factor = A[row * n + pivot] / A[pivot * n + pivot];
            for (int col = pivot; col < n; col++) {
                A[row * n + col] -= A[pivot * n + col] * factor;
            }
            b[row] -= b[pivot] * factor;
        }
    }

    pthread_exit(NULL);
}

void* backSubstitutionThread(void* threadarg) {
    ThreadData* data = (ThreadData*) threadarg;
    int tid = data->thread_id;
    REAL* A = data->A;
    REAL* b = data->b;
    REAL* x = data->x;
    int n = data->n;

    if (tid == 0) {
        for (int row = n - 1; row >= 0; row--) {
            x[row] = b[row];
            for (int col = n - 1; col > row; col--) {
                x[row] -= A[row * n + col] * x[col];
            }
            x[row] /= A[row * n + row];
        }
    }

    pthread_exit(NULL);
}

void initializeSystem(REAL* A, REAL* b, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = (REAL)rand() / RAND_MAX;
    }
    for (int i = 0; i < n; i++) {
        b[i] = (REAL)rand() / RAND_MAX;
    }
}

int main() {
    int n = 10; // Example size of the system
    REAL *A = (REAL*)malloc(n * n * sizeof(REAL));
    REAL *b = (REAL*)malloc(n * sizeof(REAL));
    REAL *x = (REAL*)malloc(n * sizeof(REAL));
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    pthread_barrier_t barrier;

    initializeSystem(A, b, n);

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    for(int i = 0; i < NUM_THREADS; i++ ) {
        thread_data[i].thread_id = i;
        thread_data[i].A = A;
        thread_data[i].b = b;
        thread_data[i].x = x;
        thread_data[i].n = n;
        thread_data[i].barrier = &barrier;
        int rc = pthread_create(&threads[i], NULL, gaussianEliminationThread, (void *)&thread_data[i]);
        if (rc) {
            fprintf(stderr, "Error: unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    for(int i = 0; i < NUM_THREADS; i++ ) {
        pthread_join(threads[i], NULL);
    }

    for(int i = 0; i < NUM_THREADS; i++ ) {
        int rc = pthread_create(&threads[i], NULL, backSubstitutionThread, (void *)&thread_data[i]);
        if (rc) {
            fprintf(stderr, "Error: unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    for(int i = 0; i < NUM_THREADS; i++ ) {
        pthread_join(threads[i], NULL);
    }

    printf("Solution Vector x: \n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    free(A);
    free(b);
    free(x);
    pthread_barrier_destroy(&barrier);

    return 0;
}
