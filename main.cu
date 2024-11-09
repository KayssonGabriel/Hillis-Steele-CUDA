#include <iostream>
#include <chrono>
#include <cuda.h>

__global__ void hillis_steele(int* array, int tamanho)
{
    int passo = 1;
    while (passo < tamanho)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index >= passo && index < tamanho)
        {
            array[index] += array[index - passo];
        }
        passo *= 2;
        __syncthreads();
    }
}

void medir_tempo_hillis_steele(int tamanho)
{
    int* h_array = new int[tamanho];
    std::fill_n(h_array, tamanho, 1);

    int* d_array;
    cudaMalloc((void**)&d_array, tamanho * sizeof(int));
    cudaMemcpy(d_array, h_array, tamanho * sizeof(int), cudaMemcpyHostToDevice);

    auto inicio = std::chrono::high_resolution_clock::now();

    int blocos = (tamanho + 255) / 256;
    hillis_steele<<<blocos, 256>>>(d_array, tamanho);
    cudaDeviceSynchronize();

    auto fim = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duracao = fim - inicio;

    std::cout << "Tamanho do array: " << tamanho
        << " | Tempo (Hillis-Steele): " << duracao.count() << " segundos" << std::endl;

    cudaMemcpy(h_array, d_array, tamanho * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    delete[] h_array;
}

int main()
{
    int tamanhos[] = {100, 1000, 10000, 100000, 1000000, 10000000};

    for (int tamanho : tamanhos)
    {
        medir_tempo_hillis_steele(tamanho);
    }

    return 0;
}
