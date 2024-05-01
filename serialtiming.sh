#!/bin/bash
#
# To run the RAJA program on the cluster:
#
#   sbatch ./run_raja.sh <matrix_size> <num_threads>


sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
serials=(1 1 1)

echo "Serial:"
for t in "${serials[@]}"; do
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./example/out/serial $s
    done
done
