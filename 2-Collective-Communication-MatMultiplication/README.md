1. Use mpicc (MPI C compiler) to compile my C program
<br> `mpicc mpi_matrix_mult.c -o mpi_matrix_mult_cc`

2. Run the Program with MPI
<br> Use `mpirun` or `mpiexec` to run the compiled program with a specified number of processes:
<br> `mpirun -np 4 ./mpi_matrix_mult_cc 64`
