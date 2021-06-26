#!/bin/bash
#SBATCH --job-name=practico2
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:00:40
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=mi@correo
#SBATCH -o practico4.out

export PATH=$PATH:/usr/local/cuda-9.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64

source /etc/profile.d/modules.sh

cd ~/

nvcc -arch=sm_60 -Xptxas -dlcm=ca main.cpp practico4.cu -o practico4 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
# nvprof ./practico4 img/fing1.pgm
nvprof --metrics shared_efficiency,gld_efficiency,gst_efficiency ./practico4 img/fing1.pgm
./practico4 img/fing1.pgm
