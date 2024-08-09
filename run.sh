#!/bin/bash
echo "Running Script Start Time : $(date)"
mamba activate carte
# Run the python script
for months in "6" "12" "18" "24"
do
    echo "Running for $months months"
    for i in "1" "3" "5" "10" "14" "15"
    do
        echo "Running for registry $i"
        python main.py --registry=$i --months=$months
    done
done