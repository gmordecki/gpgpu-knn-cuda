#include <float.h>
#include <math.h>
#include <stdio.h>
#include "csv.c"

void knn_gpu(float* dataset, int* tags, int dataset_n, int dimension, float* to_predict, int to_predict_n, int cant_tags, int k);


void exchange(float* distances, float* dataset, int* tags, int index1, int index2, int dimension) {
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

void selectSort(float* distances, float* dataset, int* tags, int length, int dimension) {
    int min;
    for(int i = 0; i < length-1; i++) {
        min = i;
        for (int j = i+1; j < length; j++) {
            if(distances[j] < distances[min]){
                min = j;
            }
        }
        if (i != min) {
            exchange(distances, dataset, tags, i, min, dimension);
        }
    }
}

float euclidean_distance(float* first_vector, float* second_vector, int dimension) {
    float squared_distance = 0;

    for (int i = 0; i < dimension ; i++) {
        squared_distance += (second_vector[i] - first_vector[i]) * (second_vector[i] - first_vector[i]);
    }

    // Lo que quiero es sólo ordenar y la raíz cuadrada no camia los órdenes por lo que puedo
    // Ahorrar tiempo obviando la cuenta
    return sqrtf(squared_distance);
    // return squared_distance;
}

int* knn_cpu(float* dataset, int* tags, int dataset_n, int dimension, float* to_predict, int to_predict_n, int cant_tags, int k) {
    // Guarda la distancia entre el vector to_predict y cada vector del dataset
    float* distances = (float *)malloc(dataset_n * sizeof(float));

    int* winner = (int *)malloc(to_predict_n * sizeof(int));
    // En cada posición i guarda la cantidad de k más cercanos que son de la clase i
    int* k_in_tag = (int *)malloc(cant_tags * sizeof(int));
    for (int to_p = 0; to_p < to_predict_n; to_p++){
        for (int tag = 0; tag < cant_tags; tag++) {
            k_in_tag[tag] = 0;
        }

        for (int i = 0; i < dataset_n; i++) {
            distances[i] = euclidean_distance(&dataset[i * dimension], &to_predict[to_p * dimension], dimension);
        }

        // printf("ANTES del sort\n");
        // for (int i = 0; i < 10 ; i++) {
        //     printf("dataset %d es ", i);
        //     for(int j = 0; j < dimension ; j++) {
        //         printf("%.1f,", dataset[i * dimension + j]);
        //     }
        //     printf(" dist es %.2f\n", distances[i]);
        // }

        selectSort(distances, dataset, tags, dataset_n, dimension);


        // printf("despues del sort\n");
        // for (int i = 0; i < 10 ; i++) {
        //     printf("dataset %d es ", i);
        //     for(int j = 0; j < dimension ; j++) {
        //         printf("%.1f,", dataset[i * dimension + j]);
        //     }
        //     printf(" dist es %.2f\n", distances[i]);
        // }

        for (int i = 0; i < k; i++) {
            // printf("k %d etiqueta %d, dato (%.1f,%.1f,%.1f,%.1f)\n", i, tags[i],
            //        dataset[i * dimension + 0],
            //        dataset[i * dimension + 1],
            //        dataset[i * dimension + 2],
            //        dataset[i * dimension + 3]
            // );
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
    }

    return winner;
}


int main(int argc, char** argv) {

    const char * path;

    // if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
    // else
    //     path = argv[argc-1];


    // Leo dataset (train)
    DataPoint *dp = Create_DataPoint();
    int dataset_n = readCSV("iris.csv", dp);
    printf("lei todo, son %d lineas\n", dataset_n);

    // Convierto el array de structs en arrays de números
    float *dataset_csv = (float *)malloc(dataset_n * DIMENSION * sizeof(float));
    int *dataset_tags = (int *)malloc(dataset_n * sizeof(int));

    for(DataPoint *loop_dp=dp->next; loop_dp!=NULL; loop_dp=loop_dp->next) {
        dataset_tags[loop_dp->id] = loop_dp->tag;
        for (int j = 0; j < DIMENSION; ++j)
            dataset_csv[loop_dp->id * DIMENSION + j] = loop_dp->vector[j];
    }

    DeleteAllDataPoint(dp);

    // Leo ejemplos a predecir (test)
    DataPoint *dp_test = Create_DataPoint();
    int to_predict_n = readCSV("test.csv", dp_test);
    printf("lei todo test, son %d lineas\n", to_predict_n);

    // Convierto el array de structs en arrays de números
    float *to_predict = (float *)malloc(to_predict_n * DIMENSION * sizeof(float));
    int *to_predict_real_tags = (int *)malloc(to_predict_n * sizeof(int));

    for(DataPoint *loop_dp=dp_test->next; loop_dp!=NULL; loop_dp=loop_dp->next) {
        to_predict_real_tags[loop_dp->id] = loop_dp->tag;
        for (int j = 0; j < DIMENSION; ++j)
            to_predict[loop_dp->id * DIMENSION + j] = loop_dp->vector[j];
    }

    DeleteAllDataPoint(dp_test);


    int* result = knn_cpu(dataset_csv, dataset_tags, dataset_n, DIMENSION, to_predict, to_predict_n, 3, 3);
    for (int i = 0; i < to_predict_n; i++)
        printf("result %d es %d y era %d\n", i, result[i], to_predict_real_tags[i]);

    return 0;
}
