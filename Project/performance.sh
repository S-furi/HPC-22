#!/bin/bash

n_cores=$(lscpu | grep -m 1 "CPU(s)" | grep -Eo "[0-9]+");

for(( i=$n_cores; $i > 0 ; i=$i-1 )); do
  elapsed=$(OMP_NUM_THREADS=${i} ./omp-sph 19999 | grep "Elapsed" | cut -d "=" -f 2)
  echo "Thread no: ${i}, time elapsed: ${elapsed}";
done;
