#!/bin/bash
sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
threads=(1 2 4 8)
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./par_gauss $s
    done
done
