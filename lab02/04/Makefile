CFLAGS=-std=c99 -Wall -Wpedantic -fopenmp

omp-merge-sort: omp-merge-sort.c
	gcc ${CFLAGS} -o omp-merge-sort omp-merge-sort.c

.PHONY: clean

clean:
	rm omp-merge-sort
