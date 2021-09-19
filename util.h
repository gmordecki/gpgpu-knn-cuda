#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
/*#include <Windows.h>*/

	#define CLK_CUEVTS_INIT \
		cudaEvent_t evt_start, evt_stop; \
		printf("Usando CUDA EVENTS para medir el tiempo\n"); \
		float t_elap; \
		cudaEventCreate(&evt_start); \
		cudaEventCreate(&evt_stop) ;

	#define CLK_CUEVTS_START \
        cudaEventRecord(evt_start, 0);

	#define CLK_CUEVTS_STOP \
        cudaEventRecord(evt_stop, 0); \
        cudaEventSynchronize(evt_stop);

	#define CLK_CUEVTS_ELAPSED \
		t_elap = (cudaEventElapsedTime( &t_elap, evt_start, evt_stop ))?0:t_elap;


	#define CLK_POSIX_INIT \
		printf("Usando gettimeofday para medir el tiempo\n"); \
		struct timeval t_i, t_f; \
		float t_elap

	#define CLK_POSIX_START \
		gettimeofday(&t_i,NULL)

	#define CLK_POSIX_STOP \
    	cudaDeviceSynchronize(); \
		gettimeofday(&t_f,NULL) 

	#define CLK_POSIX_ELAPSED \
		t_elap = ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0 - \
			 	 ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0))


void clockStart(cudaEvent_t start);
void clockStop(cudaEvent_t stop);
float clockElapsed(cudaEvent_t start, cudaEvent_t stop);
void clockInit(cudaEvent_t *start, cudaEvent_t *stop);
