#include "util.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <limits.h>
#include "device_launch_parameters.h"
#include "cuda_commons.cu"
#include "distances.cu"
#include "sorts.cu"

using namespace std;


int* knn_gpu(float* dataset, int* tags, int dataset_n, int dimension, float* to_predict,
             int to_predict_n, int cant_tags, int k, int block_size, int distance_algorithm,
             int distances_calc_algorithm, int sort_algorithm) {
    float *gpu_dataset, *gpu_to_predict, *distances_gpu;
    int *gpu_tags, *gpu_results, *gpu_winners;

    CLK_POSIX_INIT;
    CLK_POSIX_START;

    CUDA_CHK(cudaMalloc((void**)&gpu_dataset, dataset_n * dimension * sizeof(float)));
    CUDA_CHK(cudaMalloc((void**)&gpu_to_predict, to_predict_n * dimension * sizeof(float)));
    CUDA_CHK(cudaMalloc((void**)&gpu_tags, dataset_n * sizeof(int)));
    CUDA_CHK(cudaMalloc((void**)&distances_gpu, dataset_n * to_predict_n * sizeof(float)));
    CUDA_CHK(cudaMalloc((void**)&gpu_results, k * to_predict_n * sizeof(int)));
    CUDA_CHK(cudaMalloc((void**)&gpu_winners, to_predict_n * sizeof(int)));

    CUDA_CHK(cudaMemcpy(gpu_dataset, dataset, dataset_n * dimension * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(gpu_to_predict, to_predict, to_predict_n * dimension * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(gpu_tags, tags, dataset_n * sizeof(int), cudaMemcpyHostToDevice));

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;
    printf("Reserva y copia inicial de memoria: %f ms\n", t_elap);

    CLK_POSIX_START;

    dim3 tamGrid, tamBlock;
    if (distances_calc_algorithm == 1) {
        tamGrid = dim3(dataset_n / block_size, to_predict_n / block_size);
        tamBlock = dim3(block_size, block_size);
        if (dataset_n % block_size != 0) tamGrid.x += 1;
        if (to_predict_n % block_size != 0) tamGrid.y += 1;
        distances_kernel_naive <<< tamGrid, tamBlock >>> (
            gpu_dataset, gpu_to_predict, dataset_n, dimension, to_predict_n, distances_gpu, distance_algorithm
        );
    } else {
        int dim_x = dimension;
        if (dimension > 1024){
            dim_x = 1024;
        }
        int dim_y = 1024 / dim_x;
        if (dim_y == 0){
            dim_y = 1;
        }

        tamBlock = dim3(dim_x, dim_y);
        tamGrid = dim3(dataset_n / dim_x, to_predict_n / dim_y);
        if (to_predict_n % dim_y != 0) tamGrid.y += 1;
        if (dataset_n % dim_x != 0) tamGrid.x += 1;

        if (distances_calc_algorithm == 2) {
            distances_kernel_test_in_shared_naive <<< tamGrid, tamBlock, dimension * tamBlock.y * sizeof(float) >>> (
                gpu_dataset, gpu_to_predict, dataset_n, dimension, to_predict_n, distances_gpu, distance_algorithm
            );
        } else if (distances_calc_algorithm == 3) {
            distances_test_in_shared <<< tamGrid, tamBlock, dimension * tamBlock.y * sizeof(float) >>> (
                gpu_dataset, gpu_to_predict, dataset_n, dimension, to_predict_n, distances_gpu, distance_algorithm
            );
        } else {
            distances_kernel_test_in_shared_transposed <<< tamGrid, tamBlock, dimension * tamBlock.y * sizeof(float) >>> (
                gpu_dataset, gpu_to_predict, dataset_n, dimension, to_predict_n, distances_gpu, distance_algorithm
            );
        }
    }
    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;
    printf("Distancias: %f ms\n", t_elap);

    CLK_POSIX_START;

    if (sort_algorithm == 2) {
        quick_sort(gpu_tags, dataset_n, to_predict_n, k, &distances_gpu, gpu_results);
    } else if (sort_algorithm == 3) {
        quick_sort_better_pivot(gpu_tags, dataset_n, to_predict_n, k, &distances_gpu, gpu_results);
    } else if (sort_algorithm == 4) {
        quick_sort_improved(gpu_tags, dataset_n, to_predict_n, k, &distances_gpu, gpu_results);
    } else {
        int block_size_sort = 256;
        dim3 tamBlock_sort(block_size_sort, 1);
        dim3 tamGrid_sort(to_predict_n / block_size_sort, 1);
        if (to_predict_n % block_size_sort != 0)
            tamGrid_sort.x += 1;

        insertion_sort_kernel <<< tamGrid_sort, tamBlock_sort >>> (gpu_tags, dataset_n, to_predict_n, k, distances_gpu, gpu_results);
    }

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;
    printf("Ordenamiento: %f ms\n", t_elap);


    // copio los resultados a la memoria de la CPU
    int *results = (int*)malloc(k * to_predict_n * sizeof(int));
    CUDA_CHK(cudaMemcpy(results, gpu_results, k * to_predict_n * sizeof(int), cudaMemcpyDeviceToHost));

    CLK_POSIX_START;
    // max entre k y cant_tags
    int count_grid_width = k < cant_tags ? cant_tags : k;

    dim3 tamGrid_count(count_grid_width / block_size, to_predict_n / block_size);
    if (count_grid_width % block_size != 0) tamGrid_count.x += 1;
    if (to_predict_n % block_size != 0) tamGrid_count.y += 1;
    dim3 tamBlock_count(block_size, block_size);

    int shared_size = (k * to_predict_n + cant_tags) * sizeof(int);
    count_winner_kernel <<< tamGrid_count, tamBlock_count, shared_size >>> (gpu_results, gpu_winners, to_predict_n, k, cant_tags);

    CUDA_CHK(cudaDeviceSynchronize());

    int *winners = (int*)malloc(to_predict_n * sizeof(int));
    // transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(winners, gpu_winners, to_predict_n * sizeof(int), cudaMemcpyDeviceToHost));

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    // liberar la memoria
    CUDA_CHK(cudaFree(gpu_dataset));
    CUDA_CHK(cudaFree(gpu_to_predict));
    CUDA_CHK(cudaFree(gpu_tags));
    CUDA_CHK(cudaFree(distances_gpu));
    CUDA_CHK(cudaFree(gpu_results));
    CUDA_CHK(cudaFree(gpu_winners));

    return winners;
}
