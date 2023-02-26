#!/bin/bash

# The results contained in the ./raw/ folder are as follows:
# - *_results : n_particles=5000, nsteps=100
# - *_results_bg : n_particles=10000, nsteps=50

echo "p,t1,t2,t3,t4,t5"
for (( i=1; $i < 9; i=$i+1 )); do
  echo -n "${i},"
  for (( j=0; $j < 5; j=$j+1 )); do
    elapsed=$(OMP_NUM_THREADS=${i} ../../omp-sph 5000 100| grep "Elapsed" | cut -d " " -f 3)
    if (( j == 4)) ; then
      echo "${elapsed}"
    else 
      echo -n "${elapsed},"
    fi
  done
done
