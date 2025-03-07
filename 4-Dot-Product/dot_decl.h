#ifndef DOT_DECL_H
#define DOT_DECL_H

void init_vec(int *data, int size);
int dot_product_cuda(int rank, int work, int *A, int *B);
int dot_product_host(int nprocs, int n, int *x, int *y);
int sum(int size, int *data);

#endif
