#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXDIM (1 << 12) /* 4096 */
#define ROOT 0

void mat_mult(double *a, double *b, double *c, int n, int my_work);
void init_data(double *data, int data_size);
int check_result(double *C, double *D, int n);

int main(int argc, char *argv[]) {
    int n = 64, n_sq, my_work;
    int my_rank, num_procs;
    double *A, *B, *C, *D;
    double start_time, end_time, elapsed;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n > MAXDIM) n = MAXDIM;
    }
    
    n_sq = n * n;
    my_work = n / num_procs;
    
    A = (double *) malloc(sizeof(double) * n_sq);
    B = (double *) malloc(sizeof(double) * n_sq);
    C = (double *) malloc(sizeof(double) * n_sq);
    D = (double *) malloc(sizeof(double) * n_sq);
    
    if (my_rank == ROOT) {
        init_data(A, n_sq);
        init_data(B, n_sq);
    }
    
    MPI_Bcast(B, n_sq, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    
    double *local_A = (double *) malloc(sizeof(double) * my_work * n);
    double *local_C = (double *) malloc(sizeof(double) * my_work * n);
    
    if (my_rank == ROOT) {
        for (int i = 1; i < num_procs; i++) {
            MPI_Send(A + i * my_work * n, my_work * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < my_work * n; i++) {
            local_A[i] = A[i];
        }
    } else {
        MPI_Recv(local_A, my_work * n, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    start_time = MPI_Wtime();
    mat_mult(local_A, B, local_C, n, my_work);
    
    if (my_rank == ROOT) {
        for (int i = 0; i < my_work * n; i++) {
            C[i] = local_C[i];
        }
        for (int i = 1; i < num_procs; i++) {
            MPI_Recv(C + i * my_work * n, my_work * n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_C, my_work * n, MPI_DOUBLE, ROOT, 1, MPI_COMM_WORLD);
    }
    
    if (my_rank == ROOT) {
        end_time = MPI_Wtime();
        elapsed = end_time - start_time;
        mat_mult(A, B, D, n, n);
        int flag = check_result(C, D, n);
        if (flag) printf("Test: FAILED\n");
        else {
            printf("Test: PASSED\n");
            printf("Total time %f seconds.\n", elapsed);
        }
    }
    
    free(A);
    free(B);
    free(C);
    free(D);
    free(local_A);
    free(local_C);
    
    MPI_Finalize();
    return 0;
}

void mat_mult(double *a, double *b, double *c, int n, int my_work) {
    for (int i = 0; i < my_work; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void init_data(double *data, int data_size) {
    for (int i = 0; i < data_size; i++) {
        data[i] = rand() % 10;
    }
}

int check_result(double *C, double *D, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (C[i * n + j] != D[i * n + j]) {
                printf("ERROR: C[%d][%d]=%f != D[%d][%d]=%f\n", i, j, C[i * n + j], i, j, D[i * n + j]);
                return 1;
            }
        }
    }
    return 0;
}