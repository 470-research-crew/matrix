#!/bin/bash

MATRIX_FILE="matrix.txt"

# Expected solution values with two decimal accuracy
EXPECTED_X1=1.00
EXPECTED_X2=-2.00
EXPECTED_X3=-2.00

# Function to check solutions
function check_solution() {
    local x1=$(printf "%.2f" $1)
    local x2=$(printf "%.2f" $2)
    local x3=$(printf "%.2f" $3)

    if [[ $(echo "$x1 == $EXPECTED_X1" | bc -l) -eq 1 ]] && \
       [[ $(echo "$x2 == $EXPECTED_X2" | bc -l) -eq 1 ]] && \
       [[ $(echo "$x3 == $EXPECTED_X3" | bc -l) -eq 1 ]]; then
        echo "Solution is correct for $4."
    else
        echo "Solution is incorrect for $4. Got x1=$x1, x2=$x2, x3=$x3"
    fi
}


programs=(serial openmp pthread raja)

for program in "${programs[@]}"; do
    echo "Running $program in debug mode..."
    output=$(./example/out/$program -d $MATRIX_FILE)
    echo "$output"

    mapfile -t xs < <(echo "$output" | awk '/Solution x =/ { getline; print $1; getline; print $1; getline; print $1}')

    check_solution "${xs[0]}" "${xs[1]}" "${xs[2]}" "$program"
done
