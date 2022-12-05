#!/bin/bash

n_cores=$(lscpu | grep "Core(s) per socket"  | grep -Eo "[0-9]+");

for(( i=$n_cores; $i > 0 ; i=$i-1 )); do
  elapsed=$(OMP_NUM_THREADS=${i} ./omp-sph 10000 | grep "Elapsed" | cut -d "=" -f 2)
  echo "Thread no: ${i}, time elapsed: ${elapsed}";
done;
