# Check for required arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <matrix_size> <num_threads>"
    exit 1
fi

matrix_size="$1"
num_threads="$2"

export OMP_NUM_THREADS=$num_threads

srun out/pthread matrix.txt -d
