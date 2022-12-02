all: simd-matmul.c
	gcc-12 -march=native -O2 -std=c99 -Wall -Wpedantic -D_XOPEN_SOURCE=600 simd-matmul.c -o simd-matmul
