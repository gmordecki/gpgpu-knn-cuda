#include <float.h>
#include <math.h>
#include <stdio.h>
#include "util.h"
#include "csv.c"
#include "knn_cpu.cpp"


int* knn_gpu(float* dataset, int* tags, int dataset_n, int dimension, float* to_predict,
             int to_predict_n, int cant_tags, int k, int block_size, int distance_algorithm,
             int distances_calc_algorithm, int sort_algorithm);


int main(int argc, char** argv) {

    const char *train_path, *test_path;
    int dimension, precision, cant_tags, distance_algorithm, distances_calc_algorithm, sort_algorithm, k;

    if (argc < 9){
        printf("Debe ingresar los siguientess parámetros:\n"
                "1: path de archivo de train\n"
                "2: path de archivo de test\n"
                "3: dimensión de los vectores, tanto de train como de test\n"
                "4: cantidad de lugares después de la coma que tienen los números en el csv\n"
                "5: cantidad de posibles tags o, dicho de otra forma, clases\n"
                "6: K\n"
                "7: Algoritmo de distancia (1 es Manhattan, 2 es Euclídea)\n"
                "8: Algoritmo de cálculo de distancia (0 es CPU, 1 es Kernel Naive, 2 Kernel Test_in_shared_naive, 3 Kernel Test_in_shared, 4 Kernel Test_in_shared Transposed)\n"
                "9: Algoritmo de ordenamiento (0 es CPU, 1 Insertion Sort, 2 Quick Sort, 3 Quick Sort Better Pivot, 4 Quick Sort Improved)\n");
        exit(1);
    } else {
        train_path = argv[1];
        test_path = argv[2];
        dimension = atoi(argv[3]); // la dimensión de los vectores, tanto de train como de test
        precision = atoi(argv[4]); // la cantidad de lugares después de la coma que tienen los números en el csv
        cant_tags = atoi(argv[5]); // la cantidad de posibles tags o, dicho de otra forma, clases
        k = atoi(argv[6]);
        distance_algorithm = atoi(argv[7]);
        distances_calc_algorithm = atoi(argv[8]);
        sort_algorithm = atoi(argv[9]);
        if ((distances_calc_algorithm < 1 && sort_algorithm >= 1) || (distances_calc_algorithm >= 1 && sort_algorithm < 1)) {
            printf("Los algoritmos de distnacia y ordenamiento tienen que ser ambos de CPU o ambos de GPU\n");
            exit(1);
        }
    }

    // Leo dataset (train)
    DataPoint *dp = Create_DataPoint();
    int dataset_n = readCSV(train_path, dp, dimension, precision);

    // Convierto el array de structs en arrays de números
    float *dataset_csv = (float *)malloc(dataset_n * dimension * sizeof(float));
    int *dataset_tags = (int *)malloc(dataset_n * sizeof(int));

    if (distances_calc_algorithm < 4) {
        for(DataPoint *loop_dp=dp->next; loop_dp!=NULL; loop_dp=loop_dp->next) {
            dataset_tags[loop_dp->id] = loop_dp->tag;
            for (int j = 0; j < dimension; ++j) {
                dataset_csv[loop_dp->id * dimension + j] = loop_dp->vector[j];
            }
        }
    } else {
        for(DataPoint *loop_dp=dp->next; loop_dp!=NULL; loop_dp=loop_dp->next) {
            dataset_tags[loop_dp->id] = loop_dp->tag;
            for (int j = 0; j < dimension; ++j) {
                dataset_csv[j * dataset_n + loop_dp->id] = loop_dp->vector[j];
            }
        }
    }

    DeleteAllDataPoint(dp);

    // Leo ejemplos a predecir (test)
    DataPoint *dp_test = Create_DataPoint();
    int to_predict_n = readCSV(test_path, dp_test, dimension, precision);

    // Convierto el array de structs en arrays de números
    float *to_predict = (float *)malloc(to_predict_n * dimension * sizeof(float));
    int *to_predict_real_tags = (int *)malloc(to_predict_n * sizeof(int));

    for(DataPoint *loop_dp=dp_test->next; loop_dp!=NULL; loop_dp=loop_dp->next) {
        to_predict_real_tags[loop_dp->id] = loop_dp->tag;
        for (int j = 0; j < dimension; ++j)
            to_predict[loop_dp->id * dimension + j] = loop_dp->vector[j];
    }

    DeleteAllDataPoint(dp_test);

    int* result;
    if (distances_calc_algorithm < 1 && sort_algorithm < 1) {
        printf("-----------KNN CPU dist %d dist_calc %d sort %d --------------\n",
               distance_algorithm, distances_calc_algorithm, sort_algorithm);
        result = knn_cpu(dataset_csv, dataset_tags, dataset_n, dimension, to_predict, to_predict_n,
                         cant_tags, k, distance_algorithm);
    } else {
        printf("-----------KNN GPU dist %d dist_calc %d sort %d --------------\n",
               distance_algorithm, distances_calc_algorithm, sort_algorithm);
        result = knn_gpu(dataset_csv, dataset_tags, dataset_n, dimension, to_predict,
                         to_predict_n, cant_tags, k, 32, distance_algorithm,
                         distances_calc_algorithm, sort_algorithm);
    }

    float correct = 0;
    for (int i = 0; i < to_predict_n; i++) {
        if (result[i] == to_predict_real_tags[i]){
            correct++;
        }
    }
    printf("Accuracy: %.2f%\n", correct / to_predict_n * 100);

    return 0;
}
