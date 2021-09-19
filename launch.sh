#!/bin/bash
#SBATCH --job-name=practico_besteff
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --time=00:15:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=mi@correo
#SBATCH -o lab.out


export PATH=$PATH:/usr/local/cuda-10.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64

source /etc/profile.d/modules.sh

cd ~/

which nvcc
if ! nvcc --help  &> /dev/null
then
    echo "nvcc could not be found"
    ls /usr/local/
    echo "---"
    ls /usr/local/cuda-9.2
    echo "-cuda-10.1--"
    ls /usr/local/cuda-10.1/bin
    echo "-cuda-10.1--"
    ls /usr/local/cuda-10.1/
    echo "-cuda-10.2--"
    ls /usr/local/cuda-10.2/bin
    echo "-cuda-11.0--"
    ls /usr/local/cuda-11.0/bin
    echo "-cuda-9.0--"
    ls /usr/local/cuda-9.0/bin
    echo "-cuda-9.2--"
    ls /usr/local/cuda-9.2/bin
    exit
fi


nvcc -arch=sm_60 -Xptxas -dlcm=ca main.cpp lab.cu -o lab -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
# for cant_train in 100 500 1000 5000 10000 50000
# do
#     echo ""
#     echo "************** cant_train ${cant_train} **************"
#     echo ""
#     for dim in 100 500 1000 5000 10000
#     do
#         echo "------- dimension ${dim} ----------"
#         ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 0 0

#         echo ""
#         echo "-----dist 1 sort 1------------------------------------------------------------------------"
#         echo ""
#         nvprof ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 1 1
#         nvprof --metrics shared_efficiency,gld_efficiency,gst_efficiency ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 1 1
#         echo ""
#         echo "-----dist 2 sort 2------------------------------------------------------------------------"
#         echo ""
#         nvprof ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 2 2
#         nvprof --metrics shared_efficiency,gld_efficiency,gst_efficiency ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 2 2
#         echo ""
#         echo "-----dist 3 sort 3------------------------------------------------------------------------"
#         echo ""
#         nvprof ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 3 3
#         nvprof --metrics shared_efficiency,gld_efficiency,gst_efficiency ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 3 3
#         echo ""
#         echo "-----dist 4 sort 4------------------------------------------------------------------------"
#         echo ""
#         ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 4 4
#         nvprof ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 4 4
#         nvprof --metrics shared_efficiency,gld_efficiency,gst_efficiency ./lab "datasets/train_reviews_${dim}_${cant_train}.csv" "datasets/test_reviews_${dim}_${cant_train}.csv" ${dim} 5 4 5 2 4 4

#     done
#     echo ""
# done
./lab "datasets/train_reviews_100_500.csv" "datasets/test_reviews_100_500.csv" 100 5 4 5 2 0 0
./lab "datasets/train_reviews_100_500.csv" "datasets/test_reviews_100_500.csv" 100 5 4 5 2 1 1
./lab "datasets/train_reviews_100_500.csv" "datasets/test_reviews_100_500.csv" 100 5 4 5 2 2 2
./lab "datasets/train_reviews_100_500.csv" "datasets/test_reviews_100_500.csv" 100 5 4 5 2 3 3
./lab "datasets/train_reviews_100_500.csv" "datasets/test_reviews_100_500.csv" 100 5 4 5 2 4 4

# 1: path de archivo de train
# 2: path de archivo de test
# 3: dimensión de los vectores, tanto de train como de test
# 4: cantidad de lugares después de la coma que tienen los números en el csv
# 5: cantidad de posibles tags o, dicho de otra forma, clases
# 6: K
# 7: Algoritmo de distancia (1 es Manhattan, 2 es Euclídea)
# 8: Algoritmo de cálculo de distancia
#     0 es CPU,
#     1 es Kernel Naive,
#     2 Kernel Test_in_shared_naive,
#     3 Kernel Test_in_shared,
#     4 Kernel Test_in_shared Transposed
# 9: Algoritmo de ordenamiento
#     0 es CPU,
#     1 Insertion Sort,
#     2 Quick Sort,
#     3 Quick Sort Better Pivot,
#     4 Quick Sort Improved
