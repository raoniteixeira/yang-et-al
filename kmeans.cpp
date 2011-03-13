/* implementation of fingerprint segmentation method proposed in [1] 
 * using OpenCV lib.
 * 
 * shortly, the method defines two clusters (foreground and background), 
 * by considering the block-wise features CMV (coherence, median and variance) 
 * and the K-means algorithm. 
 * for addition details about this features space see, for example, [2].
 * 
 * [1] Gongping Yang et al, 'K-Means Based Fingerprint Segmentation 
 *     with Sensor Interoperability', 2010.
 * 
 * [2] Bazen and Gerez, 'Segmentation of Fingerprint Images', 2001.
 * 
 * author: raoni teixeira < rsilva at ic dot unicamp dot br >
 * 
 * start date: february, 20,  2011
 * end date: march, 12, 2011
 * */
 
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include <cfloat>

using namespace std;


//this function implements the block-wise sum of the image 'img'
IplImage* block_sum(IplImage* img, int block_size)
{
	IplImage* result = cvCloneImage(img);
	
	float sum = 0.0;
	
	for( int v = 0; v < img->height; v++ ) {		
		for( int h = 0; h < img->width; h++ ) {
			sum = 0.0;
			
			for(int by = 0; by < block_size; by++)	{
				if((v + by < img->height))	{
					float* ptr = (float*) (	img->imageData + (v+by) * img->widthStep);
					for(int bx = 0; bx < block_size; bx++)	{
						if( (h + bx < img->width)  )	{							
							sum += ptr[3*(h+bx)+0];							
						}
					}
				}
			}
			
			for(int by = 0; by < block_size; by++)	{
				if((v + by < result->height))	{
					float* ptr1 = (float*) (	result->imageData + (v+by) * result->widthStep);
					for(int bx = 0; bx < block_size; bx++)	{
						if( (h + bx < result->width)  )	{	
							
							ptr1[3*(h+bx)+0] = sum;
							ptr1[3*(h+bx)+1] = sum;
							ptr1[3*(h+bx)+2] = sum;
						}
					}
				}
			}
			
			
		}
	}
	
	return result; 
}

//these functions compute some image operations, such as square root, multiplication, 
//division, addition, subtraction, maximum and minimum. 
IplImage* sqr(IplImage * img)
{
	IplImage* result = cvCloneImage(img);
	float v1,v2;
	for( int v = 0; v < img->height; v++ ) {		
		float* ptr1 = (float*) (	img->imageData + v * img->widthStep);		
		float* ptr2 = (float*) (	result->imageData + v * result->widthStep);
		for( int h = 0; h < img->width; h++ ) {		
			v1 = ptr1[3*h+0];
			v2 = sqrt(v1);
			ptr2[3*h+0] = v2;
			ptr2[3*h+1] = v2;
			ptr2[3*h+2] = v2;
		}
	}
	
	
	return result;	
}

IplImage* multiply(IplImage* img1, IplImage *img2 )
{
	IplImage* result = cvCloneImage(img1);	
	
	float v1, v2, v3;
	
	for( int v = 0; v < img1->height; v++ ) {		
		float* ptr1 = (float*) (	img1->imageData + v * img1->widthStep);
		float* ptr2 = (float*) (	img2->imageData + v * img2->widthStep);
		float* ptr3 = (float*) (	result->imageData + v * result->widthStep);
		for( int h = 0; h < img1->width; h++ ) {		
			
			v1 = ptr1[3*h+0];
			v2 = ptr2[3*h+0];
			v3 = v1*v2;			

			ptr3[3*h+0] = v3;
			ptr3[3*h+1] = v3;
			ptr3[3*h+2] = v3;
			
		}
	}	
	
	return result; 
}

IplImage* divide(IplImage* img1, IplImage *img2 )
{
	IplImage* result = cvCloneImage(img1);	
	
	float v1, v2, v3;
	
	for( int v = 0; v < img1->height; v++ ) {		
		float* ptr1 = (float*) (	img1->imageData + v * img1->widthStep);
		float* ptr2 = (float*) (	img2->imageData + v * img2->widthStep);
		float* ptr3 = (float*) (	result->imageData + v * result->widthStep);
		for( int h = 0; h < img1->width; h++ ) {		
			v1 = ptr1[3*h+0];
			v2 = ptr2[3*h+0];			
				
			v3 = (v2 == 0.0)? 0.0 : v1/v2;
			
			ptr3[3*h+0] = v3;
			ptr3[3*h+1] = v3;
			ptr3[3*h+2] = v3;
					
		}
	}	
	
	return result; 
}

IplImage* addition(IplImage* img1, IplImage *img2 )
{
	IplImage* result = cvCloneImage(img1);	
	
	float v1, v2, v3;
	
	for( int v = 0; v < img1->height; v++ ) {		
		float* ptr1 = (float*) (	img1->imageData + v * img1->widthStep);
		float* ptr2 = (float*) (	img2->imageData + v * img2->widthStep);
		float* ptr3 = (float*) (	result->imageData + v * result->widthStep);
		for( int h = 0; h < img1->width; h++ ) {		
			v1 = ptr1[3*h+0];
			v2 = ptr2[3*h+0];
			v3 = v1+v2;
			ptr3[3*h+0] = v3;
			ptr3[3*h+1] = v3;
			ptr3[3*h+2] = v3;
					
		}
	}	
	
	return result; 
}

IplImage* subtraction(IplImage* img1, IplImage *img2 )
{
	IplImage* result = cvCloneImage(img1);	
	
	float v1, v2, v3;
	
	for( int v = 0; v < img1->height; v++ ) {		
		float* ptr1 = (float*) (	img1->imageData + v * img1->widthStep);
		float* ptr2 = (float*) (	img2->imageData + v * img2->widthStep);
		float* ptr3 = (float*) (	result->imageData + v * result->widthStep);
		for( int h = 0; h < img1->width; h++ ) {		
			v1 = ptr1[3*h+0];
			v2 = ptr2[3*h+0];
			v3 = v1-v2;
			ptr3[3*h+0] = v3;
			ptr3[3*h+1] = v3;
			ptr3[3*h+2] = v3;
					
		}
	}	
	
	return result; 
}

float maximum(IplImage *im)
{
	float max = FLT_MIN, y;
	for( int v = 0; v < im->height; v++ ) {		
		float* ptr = (float*) (	im->imageData + v * im->widthStep);
		for( int h = 0; h < im->width; h++ ) {		
			y = ptr[3*h+0];
			
			if(y > max)
				max = y;
		}
	}
	
	return max;
}

float minimum(IplImage *im)
{
	float min = FLT_MAX, y;
	for( int v = 0; v < im->height; v++ ) {		
		float* ptr = (float*) (	im->imageData + v * im->widthStep);
		for( int h = 0; h < im->width; h++ ) {		
			y = ptr[3*h+0];
			if(y < min)
				min = y;
		}
	}
	
	return min;
}

//convert matrix to grayscale image, wich contains valeus in range [0,255].
IplImage* mat2gray(IplImage* img)
{
	IplImage *result = cvCreateImage(cvSize(img->width, img->height),IPL_DEPTH_8U,3);
	float min, max, y;
	int x; 
	min = minimum(img);
	max = maximum(img);
	
	for( int v = 0; v < img->height; v++ ) {		
		float* ptr = (float*) (	img->imageData + v * img->widthStep);
		uchar* ptr1 = (uchar*) (	result->imageData + v * result->widthStep);
		for( int h = 0; h < img->width; h++ ) {		
			y = ptr[3*h+0];
			
			x = 255*((y-min)/(max-min));
			
			ptr1[3*h+0] = x;
			ptr1[3*h+1] = x;
			ptr1[3*h+2] = x;
					
		}
	}	
	
	
	return result;
}


//these functions extract, respectively, the block-wise coherence, median and variance 
IplImage* coherence(IplImage* img, int block_size)
{	
	/*horizontal and vertical components of gradient*/
	IplImage *Gx = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels);
	cvSobel(img, Gx, 1, 0, 3 );
	IplImage *Gy = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, img->nChannels); 
	cvSobel(img, Gy, 0, 1, 3 );		
	
	IplImage *G2x = multiply(Gx, Gx);		
	IplImage *Gxx = block_sum(G2x, block_size);	
	cvReleaseImage(&G2x); 
	
	IplImage *G2y = multiply(Gy, Gy);
	IplImage *Gyy = block_sum(G2y, block_size);	
	cvReleaseImage(&G2y); 
	
	IplImage *Gxy_ = multiply(Gx, Gy);
	IplImage *Gxy = block_sum(Gxy_, block_size);
	cvReleaseImage(&Gy); 
	cvReleaseImage(&Gx); 
	cvReleaseImage(&Gxy_); 
	
	/*numerator*/
	IplImage *s = subtraction(Gxx, Gyy);
	IplImage *s2 = multiply(s, s);
	cvReleaseImage(&s); 
	
	IplImage* Gxy2 = multiply(Gxy, Gxy);
	IplImage* four = cvCloneImage(Gxy2);
	
	
	CvScalar value; value.val[0]=4.0; value.val[1]=4.0; value.val[2]=4.0;
	cvSet(four, value);
	IplImage* Gxy4 = multiply(four, Gxy2);	
	IplImage* a = addition(s2, Gxy4); 	
	IplImage* numerator = sqr(a);
	cvReleaseImage(&four); 
	cvReleaseImage(&s2); 
	cvReleaseImage(&a);
	cvReleaseImage(&Gxy2);
	/*denominator*/
	IplImage* denominator = addition(Gxx, Gyy);	
	
	IplImage *d = divide(numerator, denominator);	
	
	cvReleaseImage(&Gxx);
	cvReleaseImage(&Gyy);
	cvReleaseImage(&numerator);
	cvReleaseImage(&denominator);
	cvReleaseImage(&Gxy);
	cvReleaseImage(&Gxy4);	
	
	return d;
}

IplImage* median(IplImage* im, int block_size)
{
	
	IplImage* result = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F, im->nChannels);
	
	float sum = 0.0;
	int n = 0;
	for( int v = 0; v < im->height; v++ ) {		
		for( int h = 0; h < im->width; h++ ) {
			sum = 0.0;
			n = 0;
			for(int by = 0; by < block_size; by++)	{
				if((v + by < im->height))	{
					uchar* ptr = (uchar*) (	im->imageData + (v+by) * im->widthStep);
					for(int bx = 0; bx < block_size; bx++)	{
						if( (h + bx < im->width)  )	{							
							sum += ptr[3*(h+bx)+0];							
							n++;
						}
					}
				}
			}
			
			for(int by = 0; by < block_size; by++)	{
				if((v + by < result->height))	{
					float* ptr1 = (float*) (	result->imageData + (v+by) * result->widthStep);
					for(int bx = 0; bx < block_size; bx++)	{
						if( (h + bx < result->width)  )	{	
												
							ptr1[3*(h+bx)+0] = sum/n;
							ptr1[3*(h+bx)+1] = sum/n;
							ptr1[3*(h+bx)+2] = sum/n;
						}
					}
				}
			}
			
			
		}
	}
	
	return result; 
}

IplImage* variance(IplImage* im, int block_size)
{
	
	IplImage *mean = median(im, block_size);
	IplImage* result = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F, im->nChannels);
	
	float sum = 0.0;
	int n = 0;
	for( int v = 0; v < im->height; v++ ) {		
		for( int h = 0; h < im->width; h++ ) {
			sum = 0.0;
			n = 0;
			for(int by = 0; by < block_size; by++)	{
				if((v + by < im->height))	{
					uchar* ptr = (uchar*) (	im->imageData + (v+by) * im->widthStep);
					float* ptr1 = (float*)(	mean->imageData + (v+by) * mean->widthStep);
					for(int bx = 0; bx < block_size; bx++)	{
						if( (h + bx < im->width)  )	{							
							sum += (ptr1[3*(h+bx)+0]-ptr[3*(h+bx)+0])*(ptr1[3*(h+bx)+0]-ptr[3*(h+bx)+0]);							
							n++;
						}
					}
				}
			}
			
			for(int by = 0; by < block_size; by++)	{
				if((v + by < result->height))	{
					float* ptr2 = (float*) (	result->imageData + (v+by) * result->widthStep);
					for(int bx = 0; bx < block_size; bx++)	{
						if( (h + bx < result->width)  )	{	
								
							ptr2[3*(h+bx)+0] = sum/n;
							ptr2[3*(h+bx)+1] = sum/n;
							ptr2[3*(h+bx)+2] = sum/n;
						}
					}
				}
			}
			
			
		}
	}	
	
	cvReleaseImage(&mean);
	return result;	
}

//features vector
struct Features
{
	IplImage *f; //features vector
	
	Features(IplImage *im, int block_size) 	{
		
		IplImage *coh = coherence(im, block_size);		
		IplImage *mean = median(im, block_size);
		IplImage *var = variance(im, block_size);	

		f = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_32F,3);
		
		
		for( int v = 0; v < f->height; v++ ) {					
		
			float* ptr = (float*) (	f->imageData + v * f->widthStep);								
			float* ptrC = (float*) (	coh->imageData + v * coh->widthStep);				
			float* ptrM = (float*) (	mean->imageData + v * mean->widthStep);				
			float* ptrV = (float*) (	var->imageData + v * var->widthStep);				
		
			for( int h = 0; h < f->width; h++ ) {						
				
				ptr[3*h+0] = ptrC[3*h+0];
				ptr[3*h+1] = ptrM[3*h+0];
				ptr[3*h+2] = ptrV[3*h+0];				
				
			}
			
		}
		
		cvReleaseImage(&coh);
		cvReleaseImage(&mean);
		cvReleaseImage(&var);
	}		
		~Features(){ cvReleaseImage(&f); }
};

//this function implements the segmentation method
IplImage* SKI(IplImage* im)
{
	IplImage *seg = cvCloneImage(im);
	Features MVC(im, 15);	
		
	CvMat p = cvMat( im->height*im->width,1, CV_32FC3,MVC.f->imageData);
	CvMat* pts = &p;
	CvMat* clusters = cvCreateMat( im->height*im->width, 1, CV_32SC1 );
	cvKMeans2(pts, 2, clusters, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 10.0 ) );
		
	int k = 0;
	for(int v = 0; v < seg->height; v++) {
		uchar* ptr = (uchar*) (	seg->imageData + v * seg->widthStep);			
		for(int h = 0; h < seg->width; h++, k++)	{    
			
			if(clusters->data.i[k] == 0)	{
				ptr[3*h+0] = 255;
				ptr[3*h+1] = 255;
				ptr[3*h+2] = 255;
			}else		{
				ptr[3*h+0] = 0;
				ptr[3*h+1] = 0;
				ptr[3*h+2] = 0;
			}
			
			
		}
		
	}
	cvReleaseMat(&clusters);
	return seg;
}


int main(int argc, char** argv)
{
	try{
		
		if (argc != 3)	{
			cout << "usage: " << argv[0] << " <input image> <output image>" << endl;
			return 1;
		}	
		
		IplImage *im = cvLoadImage(argv[1]);		
		IplImage *seg = SKI(im);
		
		//uncomment these lines to show the output image
		/*cvNamedWindow( "segmented image", 1); 
		cvShowImage( "segmented image", seg );
		while( cvWaitKey(15) != 27) ;
		cvDestroyWindow( "segmented image" );		*/
		
		cvSaveImage( argv[2] , seg);
		
		cvReleaseImage(&im);
		cvReleaseImage(&seg);

	}	catch( cv::Exception& e )	{
			const char* err_msg = e.what();
			cout << err_msg << endl;
	}
	
	return 0;
}
