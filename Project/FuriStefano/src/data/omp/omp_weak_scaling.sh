#!/bin/bash
N0=1500

echo "p,t1,t2,t3,t4,t5"
for (( i=1; $i < 9; i=$i+1 )); do
  PROB_SIZE=$(echo "scale=4; $N0 * sqrt($i)" | bc)
  echo -n "${i},"

  for (( j=0; $j < 5; j=$j+1 )); do
    elapsed=$(OMP_NUM_THREADS=${i} ../../omp-sph ${PROB_SIZE} | grep "Elapsed" | cut -d " " -f 3)
    if (( j == 4 )) ; then
      echo "${elapsed}"
    else 
      echo -n "${elapsed},"
    fi
  done
done

