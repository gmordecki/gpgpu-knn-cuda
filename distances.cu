

__device__ float manhattan_distance_gpu(float x, float y) {
    return fabsf(x - y);
}


__device__ float euclidean_distance_gpu(float x, float y) {
    float substraction = x - y;
    return substraction * substraction;
}


__global__ void distances_kernel_naive(float* dataset, float* to_predict, int dataset_n, int dimension,
                           int to_predict_n, float* distances, int distance_algorithm) {
    // Cada hilo en x, y guarda la distancia entre el vector x del dataset y el vector
    // y a predecir
    // distances tiene filas de to_predict_n de ancho
    // cada fila tiene todas las distancias para el to_pred_i contra todos los del dastaset
    int dataset_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (dataset_i >= dataset_n || to_pred_i >= to_predict_n)
        return;

    float distance = 0;

    for (int i = 0; i < dimension ; i++) {
        if (distance_algorithm == 1) {
            distance += manhattan_distance_gpu(
                to_predict[to_pred_i * dimension + i],
                dataset[dataset_i * dimension + i]
            );
        } else {
            distance += euclidean_distance_gpu(
                to_predict[to_pred_i * dimension + i],
                dataset[dataset_i * dimension + i]
            );
        }
    }
    distances[to_pred_i * dataset_n + dataset_i] = distance;
}


__global__ void distances_kernel_test_in_shared_naive(float* dataset, float* to_predict, int dataset_n, int dimension,
                           int to_predict_n, float* distances, int distance_algorithm) {
    // Son max(1024, dimension) hilos en la dim X
    // si es dimension cada hilo hace una distancia
    // Primero se guarda el de test en la shared memory, toda la fila hace el mismo ejemplo de test
    // Cada hilo guarda el elemento threadIdx.x del de test. Si dim > 1024 guardan los n necesarios
    // para llegar entre los 1024 hilos
    // Cada hilo en x, y guarda la distancia (al cuadrado) entre el vector x del dataset y el vector
    // y a predecir
    // distances tiene filas de to_predict_n de ancho
    // cada fila tiene todas las distancias para el to_pred_i contra todos los del dastaset
    extern __shared__ float shared_test[];

    int dataset_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (to_pred_i >= to_predict_n){
        return;
    }

    int to_calc = dimension / 1024 + 1;

    // Cargo el ejemplo de test a shared memory
    for (int i = 0; i < to_calc; i++) {
        if (threadIdx.x * to_calc + i < dimension) {
            shared_test[
                dimension * threadIdx.y + threadIdx.x * to_calc + i
            ] = to_predict[
                dimension * to_pred_i + threadIdx.x * to_calc + i
            ];
        }
    }
    __syncthreads();

    if (dataset_i >= dataset_n){
        return;
    }

    int dataset_i_to_calc;
    for (int tc = 0; tc < to_calc; tc++) {
        dataset_i_to_calc = (dataset_i * to_calc) + tc;
        if (dataset_i_to_calc < dataset_n) {
            distances[to_pred_i * dataset_n + dataset_i_to_calc] = 0;

            for (int i = 0; i < dimension; i++) {
                if (distance_algorithm == 1) {
                    distances[to_pred_i * dataset_n + dataset_i_to_calc] += manhattan_distance_gpu(
                        dataset[(dataset_i_to_calc) * dimension + i],
                        shared_test[threadIdx.y * dimension + i]
                    );
                } else {
                    distances[to_pred_i * dataset_n + dataset_i_to_calc] += euclidean_distance_gpu(
                        dataset[(dataset_i_to_calc) * dimension + i],
                        shared_test[threadIdx.y * dimension + i]
                    );
                }
            }
        }
    }
}


__global__ void distances_test_in_shared(const float* __restrict__ dataset, const float* __restrict__ to_predict,
                         int dataset_n, int dimension, int to_predict_n, float* distances, int distance_algorithm) {
    // Son max(1024, dimension) hilos en la dim X
    // si es dimension cada hilo hace una distancia
    // Primero se guarda el de test en la shared memory, toda la fila hace el mismo ejemplo de test
    // Cada hilo guarda el elemento threadIdx.x del de test. Si dim > 1024 guardan los n necesarios
    // para llegar entre los 1024 hilos
    // Cada hilo en x, y guarda la distancia (al cuadrado) entre el vector x del dataset y el vector
    // y a predecir
    // distances tiene filas de to_predict_n de ancho
    // cada fila tiene todas las distancias para el to_pred_i contra todos los del dastaset

    // va a tener dimension * tamBlock.y * sizeof(float)
    extern __shared__ float shared_test[];

    int dataset_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (to_pred_i >= to_predict_n) {
        return;
    }

    int to_calc = dimension / 1024;
    if (dimension % 1024 != 0) to_calc += 1;

    // Cargo el ejemplo de test a shared memory
    for (int i = 0; i < to_calc; i++) {
        if (i * 1024 + threadIdx.x < dimension) {
            shared_test[
                dimension * threadIdx.y + i * 1024 + threadIdx.x
            ] = to_predict[
                dimension * to_pred_i + i * 1024 + threadIdx.x
            ];
        }
    }

    __syncthreads();

    if (dataset_i >= dataset_n){
        return;
    }

    // Cada hilo tiene que, potencialmente, calcular más de una distancia (si dataset_n > 1024)
    to_calc = dataset_n / 1024;
    if (dataset_n % 1024 != 0) to_calc += 1;

    float distance;
    int dataset_i_to_calc_dist;

    for (int tc = 0; tc < to_calc; tc++) {
        dataset_i_to_calc_dist = (1024 * tc) + dataset_i;
        if (dataset_i_to_calc_dist < dataset_n) {
            distance = 0;
            for (int i = 0; i < (dimension / 32) + 1; i++) {
                for (int j = 0; j < 32; j++) {
                    if (i * 32 + ((j + threadIdx.x) % 32) < dimension) {
                        if (distance_algorithm == 1) {
                            distance += manhattan_distance_gpu(
                                dataset[(dataset_i_to_calc_dist) * dimension + i * 32 + ((j + threadIdx.x) % 32)],
                                shared_test[threadIdx.y * dimension + i * 32 + ((j + threadIdx.x) % 32)]
                            );
                        } else {
                            distance += euclidean_distance_gpu(
                                dataset[(dataset_i_to_calc_dist) * dimension + i * 32 + ((j + threadIdx.x) % 32)],
                                shared_test[threadIdx.y * dimension + i * 32 + ((j + threadIdx.x) % 32)]
                            );
                        }
                    }
                }
            }
            distances[to_pred_i * dataset_n + dataset_i_to_calc_dist] = distance;
        }
    }
}

__global__ void distances_kernel_test_in_shared_transposed(const float* __restrict__ dataset, const float* __restrict__ to_predict,
                         int dataset_n, int dimension, int to_predict_n, float* distances, int distance_algorithm) {
    // Son max(1024, dimension) hilos en la dim X
    // si es dimension cada hilo hace una distancia
    // Primero se guarda el de test en la shared memory, toda la fila hace el mismo ejemplo de test
    // Cada hilo guarda el elemento threadIdx.x del de test. Si dim > 1024 guardan los n necesarios
    // para llegar entre los 1024 hilos
    // Cada hilo en x, y guarda la distancia (al cuadrado) entre el vector x del dataset y el vector
    // y a predecir
    // distances tiene filas de to_predict_n de ancho
    // cada fila tiene todas las distancias para el to_pred_i contra todos los del dastaset
    extern __shared__ float shared_test[];

    int dataset_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (to_pred_i >= to_predict_n){
        return;
    }

    int to_calc = dimension / 1024;
    if (dimension % 1024 != 0) to_calc += 1;

    // Cargo el ejemplo de test a shared memory
    for (int i = 0; i < to_calc; i++) {
        if (i * 1024 + threadIdx.x < dimension) {
            shared_test[
                dimension * threadIdx.y + i * 1024 + threadIdx.x
            ] = to_predict[
                dimension * to_pred_i + i * 1024 + threadIdx.x
            ];
        }
    }

    __syncthreads();

    if (dataset_i >= dataset_n){
        return;
    }

    float distance;
    int dataset_i_to_calc;

    for (int tc = 0; tc < to_calc; tc++) {
        dataset_i_to_calc = (dataset_i * to_calc) + tc;
        if (dataset_i_to_calc < dataset_n) {
            distance = 0;
            distances[to_pred_i * dataset_n + dataset_i_to_calc] = 0;

            for (int i = 0; i < dimension; i++) {
                if (distance_algorithm == 1) {
                    distance += manhattan_distance_gpu(
                        dataset[i * dataset_n + dataset_i_to_calc],
                        shared_test[threadIdx.y * dimension + i]
                    );
                } else {
                    distance += euclidean_distance_gpu(
                        dataset[i * dataset_n + dataset_i_to_calc],
                        shared_test[threadIdx.y * dimension + i]
                    );
                }
            }
            distances[to_pred_i * dataset_n + dataset_i_to_calc] = distance;
        }
    }
}
