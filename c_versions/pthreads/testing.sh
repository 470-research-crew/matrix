#!/bin/bash

# Define matrix sizes and thread counts to test
sizes=(300 424 599 847 1197 1692 2392 3382 4782 6762 9562)
threads=(1 2 4 8)

# Loop over each thread count
for t in "${threads[@]}"; do
    # Loop over each matrix size
    for s in "${sizes[@]}"; do
        echo "Size: $s, Threads: $t"
        ./gaus_pthread $s $t
    done
done
