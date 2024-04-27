#!/bin/bash
#
# To run the RAJA program on the cluster:
#
#   sbatch ./run_raja.sh <matrix_size> <num_threads>

# Check for required arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <matrix_size> <num_threads>"
    exit 1
fi

matrix_size="$1"
num_threads="$2"

# Check if matrix size and num_threads are numbers
if ! [[ "$matrix_size" =~ ^[0-9]+$ ]]; then
    echo "Error: matrix size must be a positive integer."
    exit 1
fi

if ! [[ "$num_threads" =~ ^[0-9]+$ ]]; then
    echo "Error: number of threads must be a positive integer."
    exit 1
fi

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$num_threads

echo "Starting RAJA program with matrix size $matrix_size using $OMP_NUM_THREADS threads..."
srun ./example/raja $matrix_size
echo "RAJA program has completed."

echo "Starting Pthread program with matrix size $matrix_size using $OMP_NUM_THREADS threads..."
srun ./example/pthread $matrix_size
echo "Pthread program has completed."

echo "Starting OpenMP program with matrix size $matrix_size using $OMP_NUM_THREADS threads..."
srun ./example/openmp $matrix_size
echo "OpenMP program has completed."

echo "Starting Serial program with matrix size $matrix_size using $OMP_NUM_THREADS threads..."
srun ./example/serial $matrix_size
echo "Serial program has completed."