GCC=gcc-12
AUTO_SIMD_FLAGS=-march=native -ftree-vectorize -fopt-info-vec-all -funsafe-math-optimizations
C_FLAGS=-std=c99 -Wall -Wpedantic 


auto: simd-dot.c
	${GCC} -O2 ${AUTO_SIMD_FLAGS} simd-dot.c -o simd-dot 2>&1 | grep "loop vectorized"

serial: simd-dot.c
	${GCC} -O2 simd-dot.c -o simd-dot

vect: simd-dot.c
	${GCC} ${C_FLAGS} -O2 -march=native -g -ggdb simd-dot.c -o simd-dot -lm
