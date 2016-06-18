//294.605


#define _CRT_SECURE_NO_WARNINGS
#define LEVELS 256

#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <time.h>
#define pi 22/7

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    VideoCapture cap(argv[1]); 
	
    //time calculations
  	clock_t begin,end;
  	begin=clock();

    while (1) {
  
    Mat src;
	if(!cap.read(src)){
	break;
	}
	Mat rgb(src.rows, src.cols, CV_8UC3);

	float r, g, b, h, s, in , h2;
	int  size = src.rows * src.cols;
	int count[256] = {0};
	for(int i = 0; i < src.rows; i++){
      for(int j = 0; j < src.cols; j++){
          b = src.at<Vec3b>(i, j)[0];
          g = src.at<Vec3b>(i, j)[1];
          r = src.at<Vec3b>(i, j)[2];

          in = (b + g + r) / 3;// calculate intensity
		  count[(int)in] = count[(int)in] +1;
	  }
	}

	float pmf[256];  ///parallel
	for (int i=0;i<256;i++){
		pmf[i] = (float)count[i]/size;
	}

	float cdf[256] = {0};
	cdf[0] = pmf[0];
	for(int i=1;i<256;i++){
		cdf[i] = pmf[i] + cdf[i-1];
	}

	int newcdf[256] = {0}; ///parallel
	for(int i=1;i<256;i++){
		newcdf[i] =  (int)(cdf[i]*(LEVELS-1));
	}
	
	int in2;
	for (int i=0;i<src.rows;i++){
		for (int j=0;j<src.cols;j++){
			b = src.at<Vec3b>(i, j)[0];
			g = src.at<Vec3b>(i, j)[1];
			r = src.at<Vec3b>(i, j)[2];

			in = (b + g + r) / 3;// calculate intensity
			in2 = newcdf[(int)in];
			
			r = in2/in*r;
			g = in2/in*g;
			b = in2/in*b;
			
			if(r>255) r = 255; if(r<0) r = 0;
			if(g>255) g = 255; if(g<0) g = 0;
			if(b>255) b = 255; if(b<0) b = 0;
			
		  rgb.at<Vec3b>(i, j)[0] = (uchar)b;
          rgb.at<Vec3b>(i, j)[1] = (uchar)g;
          rgb.at<Vec3b>(i, j)[2] = (uchar)r;

		}
	}

		//imshow("HSI image", hsi);
		//imshow("HIST image", rgb);
    
		if(waitKey(30) == 27) {
			cout << "esc key is pressed by user" << endl; 
			break; 
		}
		
	}

	//finish time measurements
  	end=clock();
  	double cputime=(double)((end-begin)/(float)CLOCKS_PER_SEC);
  	//printf("Time using CPU for calculation is %.10f\n",cputime);


	

	cout << "The cpu time is: "<<cputime << endl;
    

		return 0;

}
