#!/bin/bash

MATRIX_FILE="matrix_.txt"

# Expected solution values with two decimal accuracy
EXPECTED_X1=3.70
EXPECTED_X2=1.60
EXPECTED_X3=2.60
EXPECTED_X4=1.10
TOLERANCE=0.02

# Function to check solutions
function check_solution() {
    local x1=$(printf "%.2f" $1)
    local x2=$(printf "%.2f" $2)
    local x3=$(printf "%.2f" $3)
    local x4=$(printf "%.2f" $4)

    if (( $(bc <<< "scale=2; sqrt(($x1 - $EXPECTED_X1)^2) < $TOLERANCE") )) && \
       (( $(bc <<< "scale=2; sqrt(($x2 - $EXPECTED_X2)^2) < $TOLERANCE") )) && \
       (( $(bc <<< "scale=2; sqrt(($x3 - $EXPECTED_X3)^2) < $TOLERANCE") )) && \
       (( $(bc <<< "scale=2; sqrt(($x4 - $EXPECTED_X4)^2) < $TOLERANCE") )); then
        echo "Solution is correct for $5."
    else
        echo "Solution is incorrect for $5. Got x1=$x1, x2=$x2, x3=$x3, x4=$x4"
    fi
}

programs=(serial openmp pthread raja cuda)

for program in "${programs[@]}"; do
    echo "Running $program in debug mode..."
    output=$(srun ./example/out/$program -d $MATRIX_FILE)
    echo "$output"

    # Extract and round the values of x from the output
    read x1 x2 x3 x4 <<< $(echo "$output" | awk '/Solution x =/ { for(i = 1; i <= 4; i++) { getline; print $1 } }' | xargs printf "%.2f %.2f %.2f %.2f")
    check_solution $x1 $x2 $x3 $x4 "$program"
done
