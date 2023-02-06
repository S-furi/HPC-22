GCC=gcc
CFLAGS=-std=c99 -Wall -Wpedantic
LIB=-fopenmp -lm


all: omp-sph

omp-sph: omp-sph.c
	${GCC} ${CFLAGS} -o omp-sph omp-sph.c ${LIB}
	

.PHONY: clean

clean:
	rm omp-sph
