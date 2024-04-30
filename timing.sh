#!/bin/bash
#
# To run the RAJA program on the cluster:
#
#   sbatch ./run_raja.sh <matrix_size> <num_threads>

sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
threads=(1 2 4 8)

echo "Serial"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/serial $s
    done
done

echo "OpenMP"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/openmp $s
    done
done

echo "Pthread"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/pthread $s
    done
done

echo "Raja"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/raja $s
    done
done

echo "Cuda"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun --gres=gpu ./example/cuda $s
    done
done