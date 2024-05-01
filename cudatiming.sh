#!/bin/bash
#
# To run the RAJA program on the cluster:
#
#   sbatch ./run_raja.sh <matrix_size> <num_threads>


sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
cudas=(1 1 1)

echo "Cuda:"
for t in "${cudas[@]}"; do
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun --gres=gpu ./example/out/cuda $s
    done
done
