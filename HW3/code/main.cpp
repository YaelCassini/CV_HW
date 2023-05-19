// Author: Li Peiyao
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <windows.h>
using namespace cv;
using namespace std;


//�Ǿֲ�����ֵ����
void localMaxFilter(Mat& in, Mat& out, int size=3)
{
	Mat dilated;
	dilate(in, dilated, Mat()); //������������Ͳ���
	if (out.empty())
	{
		out.create(in.rows, in.cols, in.type());
	}
	for (int y = 0; y < in.rows; y++)
	{
		for (int x = 0; x < in.cols; x++)
		{
			//������ص��ֵ�����Ͳ���ǰ��ͬ��֤���õ�Ϊ�ֲ�����ֵ��
			if (in.at<float>(y, x) == dilated.at<float>(y, x))
			{
				out.at<float>(y, x) = in.at<float>(y, x);
			}
			else
			{
				out.at<float>(y, x) = 0.0f;
			}
		}
	}
}


string Harris(Mat& I, double k, double thres_hold, int max_size=3)
{


	SYSTEMTIME st = { 0 };
	GetLocalTime(&st);  //��ȡ��ǰʱ�� �ɾ�ȷ��ms
	string addr_pre;
	printf("%d-%02d-%02d %02d:%02d:%02d\n",
		st.wYear,
		st.wMonth,
		st.wDay,
		st.wHour,
		st.wMinute,
		st.wSecond);
	addr_pre += to_string(st.wYear);
	addr_pre += "_";
	addr_pre += to_string(st.wMonth);
	addr_pre += "_";
	addr_pre += to_string(st.wDay);
	addr_pre += "_";
	addr_pre += to_string(st.wHour);
	addr_pre += "_";
	addr_pre += to_string(st.wMinute);
	addr_pre += "_";
	addr_pre += to_string(st.wSecond);


	//��������ʼ���˲�ģ��
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	
	Mat filter;
	filter.create(I.size(), I.type());
	//��ͼ������˲�
	cv::filter2D(I, filter, I.depth(), kernel);
	imshow("kernel", filter);


	//ת�Ҷ�ͼ
	Mat gray;
	cvtColor(filter, gray, CV_BGR2GRAY);
	//cout << gray;


	//����ͼ��X������Y�����һ��ƫ��Ix��Iy�Լ�Ix^2,Iy^y,IxIy
	Mat IxIx(gray.rows, gray.cols, CV_32FC1);
	Mat IyIy(gray.rows, gray.cols, CV_32FC1);
	Mat IxIy(gray.rows, gray.cols, CV_32FC1);

	//��ʼ��Ϊ0
	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			IxIx.at<float>(y, x) = 0;
			IyIy.at<float>(y, x) = 0;
			IxIy.at<float>(y, x) = 0;
		}
	}

	//����ƫ������Ix^2,Iy^y,IxIy
	for (int y = 1; y <= I.rows - 2; y++)
	{
		for (int x = 1; x <= I.cols - 2; x++)
		{

			float ix = (float)gray.at<uchar>(y + 1, x) - (float)gray.at<uchar>(y - 1, x);
			float iy = (float)gray.at<uchar>(y, x + 1) - (float)gray.at<uchar>(y, x - 1);

			if (ix < 0)ix = -ix;
			if (iy < 0)iy = -iy;

			IxIx.at<float>(y, x) = (float)ix * (float)ix;
			IyIy.at<float>(y, x) = (float)iy * (float)iy;
			IxIy.at<float>(y, x) = (float)ix * (float)iy;
				
		}
	}

	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			if (IxIx.at<float>(y, x) ==-0)
			{
				IxIx.at<float>(y, x) = 0;
			}
			/*if (IxIx.at<float>(y, x) < 0)
			{
				IxIx.at<float>(y, x) = -IxIx.at<float>(y, x);
			}*/
			if (IxIy.at<float>(y, x) == -0)
			{
				IxIy.at<float>(y, x) = 0;
			}
			/*if (IyIy.at<float>(y, x) < 0)
			{
				IyIy.at<float>(y, x) = -IyIy.at<float>(y, x);
			}*/
			if (IyIy.at<float>(y, x) == -0)
			{
				IyIy.at<float>(y, x) = 0;
			}
			/*if (IxIy.at<float>(y, x) < 0)
			{
				IxIy.at<float>(y, x) = -IxIy.at<float>(y, x);
			}*/
		}
	}

	//cout << IxIx;
	//cout << IyIy;
	//cout << IxIy;


	//����ͼ��X������Y�����һ��ƫ��Ix��Iy�Լ�Ix^2,Iy^2,IxIy
	Mat Sxx(gray.rows, gray.cols, CV_32FC1);
	Mat Syy(gray.rows, gray.cols, CV_32FC1);
	Mat Sxy(gray.rows, gray.cols, CV_32FC1);


	//��˹�˲�
	GaussianBlur(IxIx, Sxx, Size(5, 5), 0, 0);
	GaussianBlur(IyIy, Syy, Size(5, 5), 0, 0);
	GaussianBlur(IxIy, Sxy, Size(5, 5), 0, 0);

	//cout << Sxx;
	//cout << Syy;
	//cout << Sxy;

	//Rֵ�������С����ֵ����
	Mat R(gray.rows, gray.cols, CV_32FC1);
	Mat Max(gray.rows, gray.cols, CV_32FC1);
	Mat Min(gray.rows, gray.cols, CV_32FC1);
	//�����ӦֵRmax
	float maxResponse = 0.0f;

	//��ʼ��Ϊ0
	for (int y = 0; y < I.rows; y++)
	{
		for (int x = 0; x < I.cols; x++)
		{
			R.at<float>(y, x) = 0;
			Max.at<float>(y, x) = 0;
			Min.at<float>(y, x) = 0;
		}
	}
	
	//���������С����ֵ�Լ�harris��ӦֵR
	for (int y = 0; y <I.rows; y++)
	{
		for (int x = 0; x <I.cols; x++)
		{
			
			float aa = 0;
			float bb = 0;
			float cc = 0;
			aa = Sxx.at<float>(y, x);
			bb = Syy.at<float>(y, x);
			cc = Sxy.at<float>(y, x);

			//���㵱ǰ���ص�harris��Ӧֵ
			float t = (aa * bb - cc * cc) - k * (aa + bb) * (aa + bb);
			
			//���������С����ֵ
			float po1 = pow(1.0 * aa - bb, 2);
			float po2 = 4.0 * cc * cc;
			float sq = sqrt((double)(po2 + po1));
			float max = 0.5 * (aa + bb + sq);
			float min = 0.5 * (aa + bb - sq);


			if (t < 0)t = 0;
			//if (max < 0)max = 0;
			//if (min < 0)min = 0;
			R.at<float>(y, x) = t;
			
			Max.at<float>(y, x) = max;
			Min.at<float>(y, x) = min;

			//��¼�����Ӧֵ
			if (t > maxResponse)
				maxResponse = t;
		}
	}
	//cout << Max;
	//cout << Min;
	//cout << R;

	Mat Max1;
	//normalize(Max, Max1, 0, 255, NORM_MINMAX, CV_64FC1);
	normalize(Max, Max1, 0, 255, NORM_MINMAX);
	Mat Min1;
	//normalize(Min, Min1, 0, 255, NORM_MINMAX, CV_64FC1);
	normalize(Min, Min1, 0, 255, NORM_MINMAX);
	Mat R1;
	//normalize(R, R1, 0, 255, NORM_MINMAX, CV_64FC1);
	normalize(R, R1, 0, 32767, NORM_MINMAX);
	Mat R2;
	//normalize(R, R1, 0, 255, NORM_MINMAX, CV_64FC1);
	normalize(R, R2, 0, 255, NORM_MINMAX);

	imshow("max_normalize", Max1);
	imshow("min_normalize", Min1);
	imshow("R_normalize", R1);
	imshow("R_normalize2", R2);
	//imshow("max", Max);
	//imshow("min", Min);
	//imshow("R", R);

	imwrite(addr_pre + "_Max.png", Max);
	imwrite(addr_pre + "_Min.png", Min); 
	imwrite(addr_pre + "_R.png", R);


	Mat localMax; 
	//���оֲ������ֵ����
	localMaxFilter(R, localMax, max_size);

	//�޳����־ֲ�����ֵ
	Mat Corner=localMax;
	threshold(localMax, Corner, thres_hold * maxResponse, 255, THRESH_BINARY);
	imshow("corner", Corner);
	imwrite(addr_pre + "_Corner.png", Corner);

	//��ԭʼͼ���ϻ��ƽǵ�����
	for (int y = 0; y < Corner.rows - 1; y++)
		for (int x = 0; x < Corner.cols - 1; x++)
			if (Corner.at<float>(y, x))
				circle(I, Point2i(x, y), 2, Scalar(0, 0, 255));

	return addr_pre;
}



int main()
{
	//��һ��Ĭ�ϵ����
	VideoCapture capture(0);
	//����Ƿ�ɹ���
	if (!capture.isOpened())
		return -1;
	//int i = 0;
	//�Ƿ��нǵ���������չʾ
	bool show = false;
	while (true)
	{
		//�����������ȡͼ��
		Mat frame;
		capture >> frame;
		
		Mat src = frame.clone();
		//src = imread("5.png");
		if (!src.data) return 0;
		imshow("video", src);

		//int delay = 1000 / rate;
		int c = waitKey(30); //waitKey(delay);
		//���¿ո�����Ե�ǰ֡ͼ�����ǵ��⣬
		//�����ǰ�Ѿ��нǵ�����������չʾ,
		//�򰴿ո�ر�չʾ����
		if (c == 32)
		{
			if (!show)
			{
				//imwrite(addr_pre + "Origin.png", src);
				string addr_pre=Harris(src, 0.05, 0.001);
				imshow("result", src);
				imwrite(addr_pre + "_Result.png", src);
				imwrite(addr_pre + "_Origin.png", frame);
			}
			else
			{
				destroyWindow("corner");
				destroyWindow("max_normalize");
				destroyWindow("min_normalize");
				destroyWindow("result");
				destroyWindow("R_normalize");
				destroyWindow("R_normalize2");
				destroyWindow("kernel");
			}
			show = !show;	
		}
		else if (c == 'q' || c == 'Q'|| c==27)  //��Q/q����Esc���رս��չʾ����
		{
			destroyWindow("corner");
			destroyWindow("max_normalize");
			destroyWindow("min_normalize");
			destroyWindow("result");
			destroyWindow("R_normalize");
			destroyWindow("R_normalize2");
			destroyWindow("kernel");
			destroyWindow("result");
			return 0;
		}	
		//i++;
	}
	waitKey();
	return 0;
}


