#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include "HoughCircle.h"
#include "HoughLine.h"
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat src, src_gray;
	Mat src_edge;
	vector<Vec3f> circles;
	vector<Line> lines;

	string address;
	cout << "请输入文件名，需要后缀..." << endl;
	cin >> address;
	//读入图像
	src = imread(address, 1);
	if (!src.data)
	{
		cout << "图像不存在！" << endl;
		return -1;
	}
	int choice = 1;
	cout << "请输入检测选择(默认只检测直线)：" << endl;
	cout << "1. 仅检测直线" << endl;
	cout << "2. 仅检测圆形" << endl;
	cout << "3. 检测直线和圆形" << endl;
	cin >> choice;

	//展示源图像
	imshow("source", src);
	cout << "按任意键继续..." << endl;
	waitKey(0);


	//转换为灰度图
	cvtColor(src, src_gray, CV_BGR2GRAY);

	//高斯滤波，减弱噪声，避免检测时产生干扰
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	//展示高斯滤波结果
	imshow("guass", src_gray);
	cout << "按任意键继续..." << endl;
	waitKey(0);

	//对灰度图再做一次均值滤波
	if(address=="2.jpg") blur(src_gray, src_edge, Size(9, 9));
	else blur(src_gray, src_edge, Size(3, 3));
	imshow("mean", src_edge);
	cout << "按任意键继续..." << endl;
	waitKey(0);

	//使用Canny算子计算图像边缘并展示
	Mat edges;
	int canny_threshold = 40;
	Canny(src_edge, edges, MAX(canny_threshold / 2, 1), canny_threshold, 3);
	imshow("edges", edges);
	cout << "按任意键继续..." << endl;
	waitKey(0);


	//使用Canny算子输出的边缘图，edges作为掩码，
	//来将原图image拷贝到目标图dst中，从而显示彩色的边缘图像
	Mat image = src.clone();
	Mat dst;
	dst.create(image.size(), image.type());
	dst = Scalar::all(0);
	image.copyTo(dst, edges);
	imshow("colorful_edge", dst);
	cout << "按任意键继续..." << endl;
	waitKey(0);
	

	if (choice == 1 || choice == 3)
	{
		//hough检测直线并返回结果
		houghlines(edges, lines, 110, 1, 1);

		//在原图像上绘制检测到的直线并输出结果
		for (size_t i = 0; i < lines.size(); i++)
		{
			Point s = lines[i].start;
			Point e = lines[i].end;
			line(src, s, e, Scalar(0, 0, 255));

			printf("x1:%d, y1:%d\n", s.x, s.y);
			printf("x2:%d, y2:%d\n", e.x, e.y);
		}
	}


	if (choice == 2 || choice == 3)
	{
		//hough检测圆形并返回结果
		houghcircles(src_gray, edges, circles, 10, 30, 0, 0);

		//在原图像上绘制检测到的圆形并输出结果
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//圆心
			circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			//圆
			circle(src, center, radius, Scalar(255, 0, 0), 2, 8, 0);
			printf("x:%d, y:%d, r:%d\n", cvRound(circles[i][0]), cvRound(circles[i][1]), radius);
		}
	}
	


	//展示绘制结果后的图像
	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	imshow("Result", src);
	cout << "按任意键结束..." << endl;
	waitKey(0);

	return 0;
}