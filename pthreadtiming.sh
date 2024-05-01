#!/bin/bash
#
# To run the RAJA program on the cluster:
#
#   sbatch ./run_raja.sh <matrix_size> <num_threads>


sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
threads=(1 2 4 8)

printf "\n"
printf "\n"
echo "Pthread:"
for t in "${threads[@]}"; do
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/out/pthread $s $t
    done
done
