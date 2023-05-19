#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

#define PI 3.1415926

// ��-�ȿռ��ۼ���
struct Polar
{
    int x;
    int y;
    int count;
};
//���ڷ���ֱ�����˵�����
struct Line
{
    Point start;
    Point end;
};
bool polar_order(Polar a, Polar b);  //��������
int get_position(Mat img, int ii, int jj, int flag, int rho=1, int theta=1);  //����ˮƽ�ʹ�ֱ��ֱ�ߣ��һ�Hough�任ʱ��ʧ����Ϣ
//hough�任���ֱ��
//img������ı�Եͼ�񣨻Ҷ�ͼ��ֻ��0,255����ֵ��
//lines��������������ֱ����ʼ��
//threshold���ۼ�����ֵ��ֻ���ۼ����е�ֵ���ڸ�ֵ���ò����ԲŻᱻѡ��
//rho���Ѳ����ķֱ��ʣ�����������
//theta���Ȳ����ķֱ��ʣ�����������
void houghlines(Mat img,vector<Line>& lines, int threshold, double rho = 1, double theta = 1);