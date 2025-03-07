/* 
   CS698 GPU Cluster Programming
   Homework 4 MPI+CUDA on dot product
   10/21/2025
   Andrew Sohn
*/

#include <mpi.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include "dot_decl.h"

#define MASTER 0
#define ROOT 0
#define MAX_ORDER 20	
#define MAX_N 1<<MAX_ORDER
#define MAX_PROCS 128
#define THREADS_PER_BLOCK 1024

#define MAX_BLOCK 1024
#define MAX_THREADS_PER_BLOCK 1024

int vec_A[MAX_N], vec_B[MAX_N];

int main(int argc, char *argv[]) {
    int n = 1 << 10;
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

    if (my_rank == MASTER) {
        int vec_A[n], vec_B[n];
        init_vec(vec_A, n);
        init_vec(vec_B, n);
        MPI_Scatter(vec_A, my_work, MPI_INT, sub_A, my_work, MPI_INT, MASTER, MPI_COMM_WORLD);
        MPI_Scatter(vec_B, my_work, MPI_INT, sub_B, my_work, MPI_INT, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(nullptr, my_work, MPI_INT, sub_A, my_work, MPI_INT, MASTER, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, my_work, MPI_INT, sub_B, my_work, MPI_INT, MASTER, MPI_COMM_WORLD);
    }

    my_prod = dot_product_cuda(my_rank, my_work, sub_A, sub_B);

    MPI_Gather(&my_prod, 1, MPI_INT, all_prods, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    if (my_rank == MASTER) {
        prod_dev = sum(nprocs, all_prods);
        prod_host = dot_product_host(nprocs, n, vec_A, vec_B);
        if (prod_host == prod_dev)
            std::cout << "Test Passed! Host and Device results match." << std::endl;
        else
            std::cerr << "Test Failed! Host and Device results differ." << std::endl;
    }
    
    delete[] sub_A;
    delete[] sub_B;
    delete[] all_prods;
    MPI_Finalize();
    return 0;
}

__global__ void dot_prod_tree_reduction(int *a, int *b, int *c, int my_work, int log_n) {
    __shared__ int temp[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < my_work)
        temp[threadIdx.x] = a[tid] * b[tid];
    else
        temp[threadIdx.x] = 0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        c[blockIdx.x] = temp[0];
}
 