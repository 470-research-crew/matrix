#!/bin/bash
#
# To run the RAJA program on the cluster:
#
#   sbatch ./run_raja.sh <matrix_size> <num_threads>


sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
threads=(1 2 4 8)
serials=(1)

echo "Serial:"
for t in "${serials[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/out/serial $s
    done
done

printf "\n"
printf "\n"
echo "OpenMP:"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/out/openmp $s
    done
done

printf "\n"
printf "\n"
echo "Pthread:"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/out/pthread $s
    done
done

printf "\n"
printf "\n"
echo "Raja:"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/out/raja $s
    done
done

printf "\n"
printf "\n"
echo "Cuda:"
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun --gres=gpu ./example/out/cuda $s
    done
done
