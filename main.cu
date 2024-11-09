#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

__global__ void hillis_steele_passo(const int* origem, int* destino, int tamanho, int passo)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < tamanho)
    {
        destino[index] = origem[index];
        if (index >= passo)
        {
            destino[index] += origem[index - passo];
        }
    }
}

void verificar_cuda(cudaError_t status, const char* operacao)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operacao) + ": " + cudaGetErrorString(status));
    }
}

struct LiberarVetorCuda
{
    void operator()(int* vetor) const
    {
        cudaFree(vetor);
    }
};

std::unique_ptr<int, LiberarVetorCuda> criar_vetor_cuda(std::size_t bytes)
{
    int* vetor = nullptr;
    verificar_cuda(cudaMalloc(reinterpret_cast<void**>(&vetor), bytes), "cudaMalloc");
    return std::unique_ptr<int, LiberarVetorCuda>(vetor);
}

void medir_tempo_hillis_steele(int tamanho)
{
    std::vector<int> vetor(tamanho, 1);
    std::size_t bytes = vetor.size() * sizeof(int);
    auto origem = criar_vetor_cuda(bytes);
    auto destino = criar_vetor_cuda(bytes);
    verificar_cuda(cudaMemcpy(origem.get(), vetor.data(), bytes, cudaMemcpyHostToDevice),
                   "Cópia para a GPU");

    constexpr int threads_por_bloco = 256;
    int blocos = (tamanho + threads_por_bloco - 1) / threads_por_bloco;
    std::int64_t trabalho = 0;
    int passos = 0;
    auto inicio = std::chrono::steady_clock::now();
    for (int passo = 1; passo < tamanho; passo *= 2)
    {
        hillis_steele_passo<<<blocos, threads_por_bloco>>>(origem.get(), destino.get(), tamanho, passo);
        verificar_cuda(cudaGetLastError(), "Lançamento do kernel");
        // A stream padrão termina o passo atual antes de usar seu resultado.
        std::swap(origem, destino);
        trabalho += tamanho - passo;
        ++passos;
    }
    verificar_cuda(cudaDeviceSynchronize(), "Sincronização do kernel");
    auto fim = std::chrono::steady_clock::now();

    verificar_cuda(cudaMemcpy(vetor.data(), origem.get(), bytes, cudaMemcpyDeviceToHost),
                   "Cópia do resultado da GPU");
    for (int i = 0; i < tamanho; ++i)
    {
        if (vetor[i] != i + 1)
        {
            throw std::runtime_error("Prefixo incorreto na posição " + std::to_string(i));
        }
    }

    std::chrono::duration<double> duracao = fim - inicio;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Tamanho do array: " << tamanho
              << " | Tempo (Hillis-Steele): " << duracao.count() << " segundos"
              << " | Trabalho: " << trabalho
              << " | Passos: " << passos << '\n';
}

int main()
{
    try
    {
        int tamanhos[] = {100, 1000, 10000, 100000, 1000000, 10000000};
        for (int tamanho : tamanhos)
        {
            medir_tempo_hillis_steele(tamanho);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Erro no benchmark: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
