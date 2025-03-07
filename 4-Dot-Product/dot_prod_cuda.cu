/*
   CS698 GPU Cluster Programming
   Homework 4 MPI+CUDA on dot product
   10/21/2025
   Andrew Sohn
*/

#include <cuda_runtime.h>
#include <iostream>
#include "dot_decl.h"

#define THREADS_PER_BLOCK 1024

// CUDA Kernel for dot product using tree reduction
__global__ void dot_prod_tree_reduction(int *a, int *b, int *c, int n) {
    __shared__ int temp[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Compute partial product
    temp[threadIdx.x] = (tid < n) ? a[tid] * b[tid] : 0;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Store result from block
    if (threadIdx.x == 0)
        c[blockIdx.x] = temp[0];
}

// CUDA dot product function
int dot_product_cuda(int rank, int my_work, int *h_A, int *h_B) {
    int *d_A, *d_B, *d_C, *h_C;
    int blocks = (my_work + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    h_C = new int[blocks];
    
    cudaMalloc((void **)&d_A, my_work * sizeof(int));
    cudaMalloc((void **)&d_B, my_work * sizeof(int));
    cudaMalloc((void **)&d_C, blocks * sizeof(int));
    
    cudaMemcpy(d_A, h_A, my_work * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, my_work * sizeof(int), cudaMemcpyHostToDevice);
    
    dot_prod_tree_reduction<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, my_work);
    
    cudaMemcpy(h_C, d_C, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    int result = 0;
    for (int i = 0; i < blocks; i++) {
        result += h_C[i];
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_C;
    
    return result;
}