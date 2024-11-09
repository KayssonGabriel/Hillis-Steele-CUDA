# Hillis–Steele com CUDA

Experimento em CUDA C++ com soma de prefixos inclusiva. O programa gera vetores de inteiros preenchidos com 1, executa o kernel e informa tempo, número de somas e número de passos para tamanhos de 100 a 10 milhões de elementos.

## Como compilar e executar

Use CMake 3.29 ou superior, CUDA Toolkit com suporte a CUDA C++20, compilador hospedeiro compatível e GPU NVIDIA com driver CUDA disponível. Com Visual Studio, configure o ambiente do compilador antes de rodar o CMake.

```sh
cmake -S . -B build
cmake --build build --config Release
```

Execute `./build/HillisSteeleCUDA` com um gerador de configuração única em Linux. No PowerShell, com o gerador Visual Studio, use `./build/Release/HillisSteeleCUDA.exe`. O relatório aparece no terminal.

## Arquivos e medição

- [main.cu](main.cu) implementa o kernel, a preparação dos vetores e a conferência do resultado. A medição inclui os lançamentos e a sincronização final, e exclui a alocação e as cópias entre memória principal e GPU.
- [CMakeLists.txt](CMakeLists.txt) define o executável e o padrão CUDA C++20.

Cada passo lê de um vetor e escreve em outro. Os lançamentos sucessivos na mesma stream garantem que o próximo passo leia o resultado completo. Para o vetor preenchido com 1, o programa confere se cada posição `i` contém `i + 1` antes de relatar a medição.
