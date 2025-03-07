#include <cuda_runtime.h>
#include "dot_decl.h"
#include <iostream>

#define THREADS_PER_BLOCK 1024

// Kernel for parallel dot product computation using reduction
__global__ void dot_prod_tree_reduction(int *a, int *b, int *c, int my_work) {
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

// CUDA wrapper function
int dot_product_cuda(int my_rank, int my_work, int *h_A, int *h_B) {
    int *d_A, *d_B, *d_C, *h_C;
    int numBlocks = (my_work + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    h_C = new int[numBlocks];

    cudaMalloc((void**)&d_A, sizeof(int) * my_work);
    cudaMalloc((void**)&d_B, sizeof(int) * my_work);
    cudaMalloc((void**)&d_C, sizeof(int) * numBlocks);

    cudaMemcpy(d_A, h_A, sizeof(int) * my_work, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * my_work, cudaMemcpyHostToDevice);

    dot_prod_tree_reduction<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, my_work);

    cudaMemcpy(h_C, d_C, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);

    int cuda_prod = sum(numBlocks, h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_C;

    return cuda_prod;
}
