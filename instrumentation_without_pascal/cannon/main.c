#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

//(C = C + A * B)
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
    if (argc < 2) {
        if (rank == 0) printf("Uso: mpirun -np <p> %s <n_global>\n", argv[0]);
        return 1;
    }

    int n_global = atoi(argv[1]);
    int n_local;
    double *A, *B, *C;
    
    double t_start, t_end;
    double t_init, t_align, t_loop = 0, t_comm = 0, t_calc = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sqrt_p = (int)sqrt(size);
    if (sqrt_p * sqrt_p != size) {
        if (rank == 0) printf("Erro: O número de processos deve ser um quadrado perfeito.\n");
        MPI_Finalize();
        return 1;
    }

    // REGIAO 1
    t_start = MPI_Wtime();
    n_local = n_global / sqrt_p;
    int local_size = n_local * n_local;

    A = (double *)calloc(local_size, sizeof(double));
    B = (double *)calloc(local_size, sizeof(double));
    C = (double *)calloc(local_size, sizeof(double));

    for (int i = 0; i < local_size; i++) {
        A[i] = rank + 1.0;
        B[i] = rank + 2.0;
    }
    t_init = MPI_Wtime() - t_start;

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

    // REGIAO 2
    t_start = MPI_Wtime();
    int src, dest;
    
    // Shift A (esquerda) por 'row' posições
    MPI_Cart_shift(cart_comm, 1, my_coords[0], &src, &dest);
    MPI_Sendrecv_replace(A, local_size, MPI_DOUBLE, dest, 1, src, 1, cart_comm, &status);

    // Shift B (cima) por 'col' posições
    MPI_Cart_shift(cart_comm, 0, my_coords[1], &src, &dest);
    MPI_Sendrecv_replace(B, local_size, MPI_DOUBLE, dest, 2, src, 2, cart_comm, &status);
    t_align = MPI_Wtime() - t_start;

    // REGIAO 3
    double loop_start = MPI_Wtime();
    for (int step = 0; step < sqrt_p; step++) {
        
        // Sub-região: Cálculo
        double c_start = MPI_Wtime();
        multiply_local(A, B, C, n_local);
        t_calc += (MPI_Wtime() - c_start);

        // Sub-região: Comunicação (Shifts do passo)
        double m_start = MPI_Wtime();
        // Move A uma posição para a esquerda
        MPI_Sendrecv_replace(A, local_size, MPI_DOUBLE, left, 3, right, 3, cart_comm, &status);
        // Move B uma posição para cima
        MPI_Sendrecv_replace(B, local_size, MPI_DOUBLE, up, 4, down, 4, cart_comm, &status);
        t_comm += (MPI_Wtime() - m_start);
    }
    t_loop = MPI_Wtime() - loop_start;

    double total_time = t_init + t_align + t_loop;

    if (rank == 0) {
        printf("\n==========================================\n");
        printf("RELATÓRIO DE EXECUÇÃO\n");
        printf("Matriz: %d x %d | Processos: %d\n", n_global, n_global, size);
        printf("------------------------------------------\n");
        printf("Tempo Inicialização:   %.6f s\n", t_init);
        printf("Tempo Alinhamento:     %.6f s\n", t_align);
        printf("Tempo Loop Principal:  %.6f s\n", t_loop);
        printf("  |_ Somente Cálculo:  %.6f s\n", t_calc);
        printf("  |_ Somente Mensagens: %.6f s\n", t_comm);
        printf("------------------------------------------\n");
        printf("TEMPO TOTAL:           %.6f s\n", total_time);
        printf("==========================================\n");
    }

    free(A);
    free(B);
    free(C);
    MPI_Finalize();
    return 0;
}