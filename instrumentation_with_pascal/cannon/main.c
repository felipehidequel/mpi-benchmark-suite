#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// (C = C + A * B)
void multiply_local(double *a, double *b, double *c, int size) {
  for (int i = 0; i < size; i++) {
    for (int k = 0; k < size; k++) {
      for (int j = 0; j < size; j++) {
        c[i * size + j] += a[i * size + k] * b[k * size + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, size, sqrt_p;
  int n_global = atoi(argv[1]);
  int n_local;
  double *A, *B, *C;
  double start_time, end_time;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  sqrt_p = (int)sqrt(size);
  if (sqrt_p * sqrt_p != size) {
    if (rank == 0)
      printf("Erro: O número de processos deve ser um quadrado perfeito (4, 9, "
             "16...)\n");
    MPI_Finalize();
    return 1;
  }

  n_local = n_global / sqrt_p;
  int local_size = n_local * n_local;

  // Alocação das submatrizes locais
  A = (double *)calloc(local_size, sizeof(double));
  B = (double *)calloc(local_size, sizeof(double));
  C = (double *)calloc(local_size, sizeof(double));

  for (int i = 0; i < local_size; i++) {
    A[i] = rank + 1.0;
    B[i] = rank + 2.0;
  }

  // Criar topologia cartesiana 2D
  int dims[2] = {sqrt_p, sqrt_p};
  int periods[2] = {1, 1}; // Grade periódica (toroidal)
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

  int my_coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, my_coords);

  // Encontrar vizinhos para os shifts
  int left, right, up, down;
  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
  MPI_Cart_shift(cart_comm, 0, 1, &up, &down);

  start_time = MPI_Wtime();

  // Shift A à esquerda por 'row' posições
  int src, dest;
  MPI_Cart_shift(cart_comm, 1, my_coords[0], &src, &dest);
  MPI_Sendrecv_replace(A, local_size, MPI_DOUBLE, dest, 1, src, 1, cart_comm,
                       MPI_STATUS_IGNORE);

  // Shift B para cima por 'col' posições
  MPI_Cart_shift(cart_comm, 0, my_coords[1], &src, &dest);
  MPI_Sendrecv_replace(B, local_size, MPI_DOUBLE, dest, 2, src, 2, cart_comm,
                       MPI_STATUS_IGNORE);

  for (int step = 0; step < sqrt_p; step++) {
    multiply_local(A, B, C, n_local);

    // Move A uma posição para a esquerda
    MPI_Sendrecv_replace(A, local_size, MPI_DOUBLE, left, 3, right, 3,
                         cart_comm, MPI_STATUS_IGNORE);
    // Move B uma posição para cima
    MPI_Sendrecv_replace(B, local_size, MPI_DOUBLE, up, 4, down, 4, cart_comm,
                         MPI_STATUS_IGNORE);
  }

  end_time = MPI_Wtime();

  if (rank == 0) {
    printf("Matriz: %d x %d | Processos: %d\n", n_global, n_global, size);
    printf("Tempo de execução: %f segundos\n", end_time - start_time);
  }

  free(A);
  free(B);
  free(C);
  MPI_Finalize();
  return 0;
}