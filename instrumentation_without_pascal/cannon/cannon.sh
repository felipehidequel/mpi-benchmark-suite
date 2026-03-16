#!/bin/bash

make

OUTPUT="resultados_cannon.txt"
echo "INICIANDO BENCHMARK EM $(date)" > $OUTPUT
echo "------------------------------------------" >> $OUTPUT

TAMANHOS=(2048 2560 3072 3584)

PROCESSOS=(4 16 36 64)

for NP in "${PROCESSOS[@]}"
do
    echo "Testando com $NP processos..."
    echo "### CONFIGURAÇÃO: $NP PROCESSOS ###" >> $OUTPUT
    
    for N in "${TAMANHOS[@]}"
    do
        echo "  -> Matriz $N x $N"
        echo "Tamanho da Matriz: $N" >> $OUTPUT
        mpirun --oversubscribe -np $NP ./main $N >> $OUTPUT
        echo "------------------------------------------" >> $OUTPUT
    done
done

echo "Benchmark finalizado! Resultados salvos em $OUTPUT"
