#include "dot_decl.h"
#include <cstdlib>

// Initializes a vector with random values (between 0 and 15)
void init_vec(int *data, int size) {
    for (int i = 0; i < size; i++)
        data[i] = rand() & 0xf;
}

// Computes sum of an integer array
int sum(int size, int *data) {
    int accum = 0;
    for (int i = 0; i < size; i++) {
        accum += data[i];
    }
    return accum;
}

// Computes dot product on CPU
int dot_product_host(int nprocs, int n, int *x, int *y) {
    int prod = 0;
    for (int i = 0; i < n; i++) {
        prod += x[i] * y[i];
    }
    return prod;
}
