
void swap(float* distances, float* dataset, int* tags, int index1, int index2, int dimension) {
    float* tmp = (float *)malloc(dimension * sizeof(float));
    for(int i = 0; i < dimension; i++)
        tmp[i] = dataset[index1 * dimension + i];
    for(int i = 0; i < dimension; i++) {
        dataset[index1 * dimension + i] = dataset[index2 * dimension + i];
        dataset[index2 * dimension + i] = tmp[i];
    }

    int tmp_tag = (int)tags[index1];
    tags[index1] = tags[index2];
    tags[index2] = tmp_tag;

    float tmp_dst = (float)distances[index1];
    distances[index1] = distances[index2];
    distances[index2] = tmp_dst;
}

void selectSort(float* distances, float* dataset, int* tags, int length, int dimension, int k) {
    int min;

    for(int i = 0; i < k; i++) {
        min = i;
        for (int j = i+1; j < length; j++) {
            if(distances[j] < distances[min]){
                min = j;
            }
        }
        if (i != min) {
            swap(distances, dataset, tags, i, min, dimension);
        }
    }
}


float manhattan_distance(float* first_vector, float* second_vector, int dimension) {
    float distance = 0;

    for (int i = 0; i < dimension ; i++) {
        distance += abs(second_vector[i] - first_vector[i]);
    }

    return distance;
}


float euclidean_distance(float* first_vector, float* second_vector, int dimension) {
    float squared_distance = 0;
    float substraction;

    for (int i = 0; i < dimension ; i++) {
        substraction = second_vector[i] - first_vector[i];
        squared_distance += substraction * substraction;
    }

    // Lo que quiero es sólo ordenar y la raíz cuadrada no cambia los órdenes por lo que puedo
    // Ahorrar tiempo obviando la cuenta
    // return sqrtf(squared_distance);
    return squared_distance;
}

int* knn_cpu(float* dataset, int* tags, int dataset_n, int dimension, float* to_predict,
             int to_predict_n, int cant_tags, int k, int distance_algorithm) {

    // Guarda la distancia entre el vector to_predict y cada vector del dataset
    float* distances = (float *)malloc(dataset_n * sizeof(float));

    int* winner = (int *)malloc(to_predict_n * sizeof(int));
    // En cada posición i guarda la cantidad de k más cercanos que son de la clase i
    int* k_in_tag = (int *)malloc(cant_tags * sizeof(int));

    float t_distances = 0;
    float t_sort = 0;
    float t_winner = 0;
    float t_elap;
    struct timeval t_i, t_f;

    for (int to_p = 0; to_p < to_predict_n; to_p++){
        for (int tag = 0; tag < cant_tags; tag++) {
            k_in_tag[tag] = 0;
        }

        gettimeofday(&t_i,NULL);
        for (int n = 0; n < dataset_n; n++) {
            if (distance_algorithm == 1) {
                distances[n] = manhattan_distance(&dataset[n * dimension], &to_predict[to_p * dimension], dimension);
            } else {
                distances[n] = euclidean_distance(&dataset[n * dimension], &to_predict[to_p * dimension], dimension);
            }
        }
        gettimeofday(&t_f,NULL);
        CLK_POSIX_ELAPSED;
        t_distances += t_elap;

        gettimeofday(&t_i,NULL);

        selectSort(distances, dataset, tags, dataset_n, dimension, k);

        gettimeofday(&t_f,NULL);
        CLK_POSIX_ELAPSED;
        t_sort += t_elap;

        gettimeofday(&t_i,NULL);
        for (int i = 0; i < k; i++) {
            k_in_tag[tags[i]] += 1;
        }

        winner[to_p] = 0;
        int max = 0;

        for (int tag = 0; tag < cant_tags; tag++) {
            if (k_in_tag[tag] > max) {
                max = k_in_tag[tag];
                winner[to_p] = tag;
            }
        }
        gettimeofday(&t_f,NULL);
        CLK_POSIX_ELAPSED;
        t_winner += t_elap;
    }

    printf("CPU Distancias: %f ms\n", t_distances);
    printf("CPU Ordenamiento: %f ms\n", t_sort);
    printf("CPU Conteo de ganadores: %f ms\n", t_winner);
    return winner;
}
