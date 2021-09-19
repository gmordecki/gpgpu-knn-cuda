all:
	nvcc -arch=sm_60 -Xptxas -dlcm=ca main.cpp lab.cu -o lab -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
