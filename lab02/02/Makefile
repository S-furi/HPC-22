CFLAGS=-fopenmp -Wall -Wpedantic
STD=-std=c99

omp-c-ray: omp-c-ray.c
	gcc ${STD} ${CFLAGS} -o omp-c-ray omp-c-ray.c -lm


sphfract.small: omp-c-ray
	./omp-c-ray -s 800x600 <sphfract.small.in> img.ppm

sphfract.big: omp-c-ray
	./omp-c-ray -s 800x600 <sphfract.big.in> img_big.ppm

dna: omp-c-ray
	./omp-c-ray -s 800x600 <dna.in> dna.ppm

spheres: omp-c-ray
	./omp-c-ray -s 800x600 <spheres.in> spheres.ppm

.PHONY: clean

clean:
	rm -rf img* *.ppm omp-c-ray

