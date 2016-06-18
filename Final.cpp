/*-------------------------------------------------CO227 Project------------------------------------------------
----------------------------------------Accelerating video enhancement with GPU---------------------------------
---------------------------------------------------Group No 8---------------------------------------------------
-----------------------------------------------E/12/047 and E/12/161------------------------------------------*/
#define _CRT_SECURE_NO_WARNINGS
#define LEVELS 256
#define pi 22/7
#define SIZE 1000

#include <stdlib.h>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "helpers.cuh"

using namespace cv;
using namespace std;

__global__ void getpmf(float * pmf_cuda, int * count_cuda, int size){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<LEVELS){
		pmf_cuda[i] = (float)count_cuda[i]/size;
	}
}

__global__ void getcdf(float * newcdf_cuda, float * cdf_cuda){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < LEVELS){
		newcdf_cuda[i] = cdf_cuda[i]*(LEVELS-1);
	}
}

__global__ void getintensity(float * intensity, float * data, int size){
	int row;
	row = blockIdx.x*blockDim.x + threadIdx.x;
		if(row<size){
			intensity[row] = (data[3*row]+data[3*row+1]+data[3*row+2])/3;	
		}
}

__global__ void editvals(float * data, float * intensity, float * newcdf, int size){
	float in,in2;
	int row;
	row = blockIdx.x*blockDim.x + threadIdx.x;
	if(row<size){
		in = intensity[row];
		in2 = newcdf[(int)in];
		
		data[3*row] = in2/in*data[3*row];
		data[3*row+1] = in2/in*data[3*row+1];
		data[3*row+2] = in2/in*data[3*row+2];
		
		if(data[3*row]>255) data[3*row] = 255; if(data[3*row]<0) data[3*row] = 0;
		if(data[3*row+1]>255) data[3*row+1] = 255; if(data[3*row+1]<0) data[3*row+1] = 0;
		if(data[3*row+2]>255) data[3*row+2] = 255; if(data[3*row+2]<0) data[3*row+2] = 0;
	}
}

int main(int argc, char* argv[])
{
    VideoCapture cap(argv[1]); 
	

    while (1) {
  
    Mat src;
	cap.read(src);
	Mat rgb(src.rows, src.cols, CV_8UC3);
	
	int count[LEVELS];
	float pmf[LEVELS];  
	float cdf[LEVELS];
	
	for(int i = 0; i<LEVELS; i++){
		count[i] = 0;
		pmf[i] = 0;
		cdf[i] = 0;
	}

	int  size = src.rows * src.cols;
	
	//pointers for cuda memory locations
	float * d_data;
	float * d_intensity;
	int * d_count;
	float * d_pmf;
	float * d_cdf;
	float * d_newcdf;
	
	//thread configuration 
	int numofblocks = (ceil(size/(float)SIZE));
	
	float * data = (float *)malloc(3*sizeof(float)*size);
	float * intensity = (float *)malloc(sizeof(float)*size);
	
// store frame details in data
	for(int i = 0; i < src.rows; i++){
      for(int j = 0; j < src.cols; j++){
          data[i*src.cols*3 + j*3]  = src.at<Vec3b>(i, j)[0];
          data[i*src.cols*3 + j*3 +1]  = src.at<Vec3b>(i, j)[1];
          data[i*src.cols*3 + j*3 + 2]  = src.at<Vec3b>(i, j)[2];
	  }
	}

//calculate intensity
	//allocate memory in cuda
	cudaMalloc((void **)&d_data,3*sizeof(float)*size); checkCudaError();
	cudaMalloc((void **)&d_intensity,sizeof(float)*size); checkCudaError();
	
	//copy contents from ram to cuda
	cudaMemcpy( d_data,data, 3*sizeof(float)*size, cudaMemcpyHostToDevice); checkCudaError();
	
	getintensity<<<numofblocks,SIZE>>>(d_intensity,d_data,size); checkCudaError();
	
	//copy the answer back
	cudaMemcpy(intensity, d_intensity, sizeof(float) * size, cudaMemcpyDeviceToHost); checkCudaError();
	
//calculate count	
	for(int i =0; i<size; i++){
		count[(int)(intensity[i])] = count[(int)(intensity[i])] +1;
	}
	
//calculate pmf
	
	//allocate memory in cuda
	cudaMalloc((void **)&d_count,sizeof(int)*LEVELS); checkCudaError();
	cudaMalloc((void **)&d_pmf,sizeof(float)*LEVELS); checkCudaError();
	
	//copy contents from ram to cuda
	cudaMemcpy( d_count,count, sizeof(int)*LEVELS, cudaMemcpyHostToDevice); checkCudaError();
	
	getpmf<<<numofblocks,SIZE>>>(d_pmf,d_count,size); checkCudaError();
	
	//copy the answer back
	cudaMemcpy(pmf, d_pmf, sizeof(float)*LEVELS, cudaMemcpyDeviceToHost); checkCudaError();
	
	cudaFree(d_count); checkCudaError();
	cudaFree(d_pmf); checkCudaError();

//calculate cdf	
	cdf[0] = pmf[0];
	for(int i=1;i<LEVELS;i++){
		cdf[i] = pmf[i] + cdf[i-1];
	}

//calculate newcdf
	//allocate memory in cuda
	cudaMalloc((void **)&d_cdf,sizeof(float)*LEVELS); checkCudaError();
	cudaMalloc((void **)&d_newcdf,sizeof(float)*LEVELS); checkCudaError();
	
	//copy contents from ram to cuda
	cudaMemcpy(d_cdf,cdf, sizeof(float)*LEVELS, cudaMemcpyHostToDevice); checkCudaError();

	getcdf<<<numofblocks,SIZE>>>(d_newcdf,d_cdf); checkCudaError();
	
	cudaFree(d_cdf); checkCudaError();
	
//edit intensity values

	editvals<<<numofblocks,SIZE>>>(d_data, d_intensity, d_newcdf, size); checkCudaError();
	
	//copy the answer back
	cudaMemcpy(data, d_data, 3*sizeof(float)*src.rows * src.cols, cudaMemcpyDeviceToHost); checkCudaError();
	
	//free
	cudaFree(d_data); checkCudaError();
	cudaFree(d_newcdf); checkCudaError();	
	cudaFree(d_intensity); checkCudaError();

// add new values to the original frame	
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
		  rgb.at<Vec3b>(i, j)[0] = (uchar)data[i*src.cols*3 + j*3];
          rgb.at<Vec3b>(i, j)[1] = (uchar)data[i*src.cols*3 + j*3 +1];
          rgb.at<Vec3b>(i, j)[2] = (uchar)data[i*src.cols*3 + j*3 +2];

		}
	}	
	
		free(data);
		free(intensity);

		imshow("Enhanced_video", rgb);
    
		if(waitKey(30) == 27) {
			cout << "esc key is pressed by user" << endl; 
			break; 
		}
		
	}
    
	return 0;

}