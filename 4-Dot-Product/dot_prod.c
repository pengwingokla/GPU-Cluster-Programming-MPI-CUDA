/* 
   CS698 GPU Cluster Programming
   Homework 4 MPI+CUDA on dot product
   10/21/2025
   Andrew Sohn
*/

#include <mpi.h>
#include <iostream>
#include <sys/time.h>
#include "dot_decl.h"

#define MASTER 0
#define MAX_ORDER 20
#define MAX_N (1 << MAX_ORDER)
#define MAX_PROCS 128

using namespace std;

int vec_A[MAX_N], vec_B[MAX_N];

int main(int argc, char *argv[]) {
    int n = 1 << 10;  // Default size of vectors
    int my_rank, nprocs;
    int my_work;
    int my_prod = 0, prod_dev = 0, prod_host = 0;
    int *sub_A, *sub_B, *all_prods;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    my_work = n / nprocs;
    sub_A = new int[my_work];
    sub_B = new int[my_work];
    all_prods = new int[nprocs];

    // Master process initializes vectors and distributes data
    if (my_rank == MASTER) {
        init_vec(vec_A, n);
        init_vec(vec_B, n);
    }

    // Scatter data among processes
    MPI_Scatter(vec_A, my_work, MPI_INT, sub_A, my_work, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(vec_B, my_work, MPI_INT, sub_B, my_work, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Compute partial dot product on each process using CUDA
    my_prod = dot_product_cuda(my_rank, my_work, sub_A, sub_B);

    // Gather results from all processes
    MPI_Gather(&my_prod, 1, MPI_INT, all_prods, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    // Master process computes final dot product and verifies correctness
    if (my_rank == MASTER) {
        prod_dev = sum(nprocs, all_prods);
        prod_host = dot_product_host(nprocs, n, vec_A, vec_B);

        if (prod_host == prod_dev)
            cout << "Test Passed! Host and Device results match." << endl;
        else
            cerr << "Test Failed! Host and Device results differ." << endl;
    }
    
    delete[] sub_A;
    delete[] sub_B;
    delete[] all_prods;
    
    MPI_Finalize();
    return 0;
}
