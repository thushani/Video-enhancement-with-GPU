//169.837

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

__global__ void getintensity(float * intensity, float * data, int rows, int cols){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
		if(row<rows && col<cols){
			int post = row*cols + col;
			intensity[post] = (data[3*post]+data[3*post+1]+data[3*post+2])/3;	
		}
}

__global__ void editvals(float * data, float * intensity, float * newcdf, int rows, int cols){
	float in,in2;
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if(row<rows && col<cols){
		int post = row*cols + col;
		in = intensity[post];
		in2 = newcdf[(int)in];

		float val = data[3*post];		
		val = in2/in*val;
		if(val>255) val = 255; if(val<0) val = 0;
		data[3*post] = val;

		val = data[3*post+1];
		val = in2/in*val;
		if(val>255) val = 255; if(val<0) val = 0;
		data[3*post+1] = val;

		val = data[3*post+2];
		val = in2/in*val;
		if(val>255) val = 255; if(val<0) val = 0;
		data[3*post+2] = val;
	}
}

int main(int argc, char* argv[])
{
    VideoCapture cap(argv[1]); 

	//start meauring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	int count[LEVELS];
	float pmf[LEVELS];  
	float cdf[LEVELS];

	//pointers for cuda memory locations
	float * d_data;
	float * d_intensity;
	float * d_cdf;

	Mat src;
	cap.read(src);
	int  size = src.rows * src.cols;

	//allocate memory in cuda
	cudaMalloc((void **)&d_data,3*sizeof(float)*size); checkCudaError();
	cudaMalloc((void **)&d_intensity,sizeof(float)*size); checkCudaError();
	cudaMalloc((void **)&d_cdf,sizeof(float)*LEVELS); checkCudaError();

	//thread configuration 
	int numofblocks = (ceil(size/(float)SIZE)); 
	dim3 numBlocks(ceil(src.cols/(float)16),ceil(src.rows/(float)16));
	dim3 threadsPerBlock(16,16);
	

    while (1) {
  
       	if(!cap.read(src)){
	break;
	}

	Mat rgb(src.rows, src.cols, CV_8UC3);
		
	for(int i = 0; i<LEVELS; i++){
		count[i] = 0;
		pmf[i] = 0;
		cdf[i] = 0;
	}

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
	
	//copy contents from ram to cuda
	cudaMemcpy( d_data,data, 3*sizeof(float)*size, cudaMemcpyHostToDevice); checkCudaError();
	
	getintensity<<<numBlocks,threadsPerBlock>>>(d_intensity,d_data,src.rows, src.cols); checkCudaError();
	
	//copy the answer back
	cudaMemcpy(intensity, d_intensity, sizeof(float) * size, cudaMemcpyDeviceToHost); checkCudaError();
	
//calculate count	
	for(int i =0; i<size; i++){
		count[(int)(intensity[i])] = count[(int)(intensity[i])] +1;
	}
	
//calculate pmf
	int mul = LEVELS-1;
	for (int i=0;i<256;i++){
		pmf[i] = (float)count[i]/size*mul;
	}

//calculate cdf	
	cdf[0] = pmf[0];
	for(int i=1;i<LEVELS;i++){
		cdf[i] = pmf[i] + cdf[i-1];
	}


//edit intensity values
	//copy contents from ram to cuda
	cudaMemcpy( d_cdf,cdf, sizeof(float)*LEVELS, cudaMemcpyHostToDevice); checkCudaError();

	editvals<<<numBlocks,threadsPerBlock>>>(d_data, d_intensity, d_cdf,src.rows, src.cols); checkCudaError();
	
	//copy the answer back
	cudaMemcpy(data, d_data, 3*sizeof(float)*src.rows * src.cols, cudaMemcpyDeviceToHost); checkCudaError();

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

	//free
	//free
	cudaFree(d_data); checkCudaError();
	cudaFree(d_cdf); checkCudaError();	
	cudaFree(d_intensity); checkCudaError();

	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);

	

	cout << elapsedtime/(float)1000 << endl;
    
	return 0;

}
