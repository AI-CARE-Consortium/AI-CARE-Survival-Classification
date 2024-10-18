#!/bin/bash
echo "Running Script Start Time : $(date)"
starttime=$(date)
#mamba activate carte
# Run the python script
for months in "6" "12" "18" "24"
do
    echo "Running for $months months"
    for i in "all" "1" "2" "3" "5" "10" "14" "15"
    do
        echo "Running for registry $i"
        python main.py --registry=$i --months=$months --inverse --dummy
        python main.py --registry=$i --months=$months --inverse
    done
done;