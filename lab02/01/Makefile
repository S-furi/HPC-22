CFLAGS=-fopenmp -Wall -Wpedantic
STD=-std=c99

all: omp-dynamic omp-dynamic-v2

omp-dynamic: omp-dynamic.c
	gcc ${STD} ${CFLAGS} -o omp-dynamic omp-dynamic.c

omp-dynamic-v2: omp-dynamic-v2.c
	gcc ${STD} ${CFLAGS} -o omp-dynamic-v2 omp-dynamic-v2.c


.PHONY: clean

clean:
	rm -rf omp-dynamic omp-dynamic-v2

