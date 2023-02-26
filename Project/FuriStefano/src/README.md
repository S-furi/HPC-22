# Progetto High Performance Computing A.A. 2022/2023 
## Contenuto 
Nella seguente cartella sono stati consegnati i codici sorgenti delle versioni
parallele dell'algoritmo SPH parallelizzate mediante **OpenMP** e **MPI**, la
**relazione** del progetto, e i risultati ottenuti dai test (in formato *csv*)
per poter verificare i calcoli effettuati, con i relativi *Jupyter Notebook*
che hanno permesso la loro manipolazione e rappresentazione grafica.

Sono inoltre pervenuti gli script per effettuare le multiple esecuzioni sia per
lo speedup/strong scaling efficiency, sia per la weak scaling efficiency,
formattando direttamente l'output in *csv*.

Gli script e i *Notebook* sono contenuti nella directory `data/`, mentre i 
risultati dei test sono contenuti nella directory `data/raw/`.

## Istruzioni 
All'interno della cartella `src/` sono contenuti i sorgenti dellaversione MPI
e OpenMP, il file `hpc.h` e il `Makefile`.

Per compilare i sorgenti è sufficiente digitare il comando `make` all'interno
della cartella `src/`: in questo modo verranno compilate entrambe le versioni.
È possibile compilare individualemente le due versioni rispettivamente `make
omp-sph` per la versione OpenMP e `make mpi-sph` per la versione MPI.

Ora è possibile eseguire i programmi con il numero desiderato di processi
mediante ``` OMP_NUM_THREADS=N ./omp-sph [n_particles] [nsteps] ``` e ```
mpirun -n N ./mpi-sph [n_particles] [nsteps] ```

Con `make clean` è possibile eliminare i sorgenti compilati dalle istruzioni
make sopracitate.
