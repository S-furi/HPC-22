CFLAGS=-fopenmp -Wall -Wpedantic
STD=-std=c99

omp-cat-map: omp-cat-map.c
	gcc ${STD} ${CFLAGS} -o omp-cat-map omp-cat-map.c

sol: omp-cat-map.c cat1368.pgm
	./omp-cat-map 36 < cat1368.pgm > sol_cat1368.pgm

.PHONY: clean

clean:
	rm -rf omp-cat-map sol*

