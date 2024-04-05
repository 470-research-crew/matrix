#!/bin/bash

sizes=(1000 1414 2000 2828 4000 5657 8000 11314 16000 22627)

threads=(1 2 4 8)
for t in "${threads[@]}"; do
    export OMP_NUM_THREADS=$t
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        srun ./par_gauss -t $s
    done
done
