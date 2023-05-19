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
	cout << "�������ļ�������Ҫ��׺..." << endl;
	cin >> address;
	//����ͼ��
	src = imread(address, 1);
	if (!src.data)
	{
		cout << "ͼ�񲻴��ڣ�" << endl;
		return -1;
	}
	int choice = 1;
	cout << "��������ѡ��(Ĭ��ֻ���ֱ��)��" << endl;
	cout << "1. �����ֱ��" << endl;
	cout << "2. �����Բ��" << endl;
	cout << "3. ���ֱ�ߺ�Բ��" << endl;
	cin >> choice;

	//չʾԴͼ��
	imshow("source", src);
	cout << "�����������..." << endl;
	waitKey(0);


	//ת��Ϊ�Ҷ�ͼ
	cvtColor(src, src_gray, CV_BGR2GRAY);

	//��˹�˲�������������������ʱ��������
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	//չʾ��˹�˲����
	imshow("guass", src_gray);
	cout << "�����������..." << endl;
	waitKey(0);

	//�ԻҶ�ͼ����һ�ξ�ֵ�˲�
	if(address=="2.jpg") blur(src_gray, src_edge, Size(9, 9));
	else blur(src_gray, src_edge, Size(3, 3));
	imshow("mean", src_edge);
	cout << "�����������..." << endl;
	waitKey(0);

	//ʹ��Canny���Ӽ���ͼ���Ե��չʾ
	Mat edges;
	int canny_threshold = 40;
	Canny(src_edge, edges, MAX(canny_threshold / 2, 1), canny_threshold, 3);
	imshow("edges", edges);
	cout << "�����������..." << endl;
	waitKey(0);


	//ʹ��Canny��������ı�Եͼ��edges��Ϊ���룬
	//����ԭͼimage������Ŀ��ͼdst�У��Ӷ���ʾ��ɫ�ı�Եͼ��
	Mat image = src.clone();
	Mat dst;
	dst.create(image.size(), image.type());
	dst = Scalar::all(0);
	image.copyTo(dst, edges);
	imshow("colorful_edge", dst);
	cout << "�����������..." << endl;
	waitKey(0);
	

	if (choice == 1 || choice == 3)
	{
		//hough���ֱ�߲����ؽ��
		houghlines(edges, lines, 110, 1, 1);

		//��ԭͼ���ϻ��Ƽ�⵽��ֱ�߲�������
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
		//hough���Բ�β����ؽ��
		houghcircles(src_gray, edges, circles, 10, 30, 0, 0);

		//��ԭͼ���ϻ��Ƽ�⵽��Բ�β�������
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//Բ��
			circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			//Բ
			circle(src, center, radius, Scalar(255, 0, 0), 2, 8, 0);
			printf("x:%d, y:%d, r:%d\n", cvRound(circles[i][0]), cvRound(circles[i][1]), radius);
		}
	}
	


	//չʾ���ƽ�����ͼ��
	namedWindow("Result", CV_WINDOW_AUTOSIZE);
	imshow("Result", src);
	cout << "�����������..." << endl;
	waitKey(0);

	return 0;
}