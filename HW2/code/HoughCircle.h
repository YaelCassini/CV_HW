#pragma once

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

#define PI 3.14159265358979
#define HOUGH_CIRCLE_RADIUS_MIN						10  //Բ��С�뾶
#define HOUGH_CIRCLE_RADIUS_MIN_DIST				2  //ͬ��Բ�����뾶��С��
#define HOUGH_CIRCLE_INTEGRITY_DEGREE				0.6  //�����ж�Բ���ϵĵ��Ƿ��㹻�ࣨ�Ƿ��ܳ�Բ��
#define HOUGH_CIRCLE_SAMEDIRECT_DEGREE				0.99  //�����ݶȼ��
#define HOUGH_CIRCLE_GRADIENT_INTEGRITY_DEGREE		0.9   //�����ݶȼ��

//��¼Բ��
struct Centers2
{
    int x;
    int y;
    int count;
};
//��������
bool center_order(Centers2 a, Centers2 b);

//��¼��������Բ�ĵľ�����ڻ�
struct Radius
{
    double dist2;
    double inner_product;
};
//��������
bool radius_order(Radius a, Radius b);
//���ڹ�һ��Բ�ļ��ͼ�񲢷Ŵ����
float normalization(float x);

//hough�ݶȷ����Բ��
//src_gray���Ҷ�ͼ��
//edges����Եͼ��
//circles��������Բ����
//min_dist����Բ��֮�����С���룬С�ڸþ�����Բ�Ľ����ϲ�
//add_threshold����ֵ���ۼ����д��ڸ�ֵ�ĵ���ܱ�ѡ��ΪԲ��
//min_Radius��Բ��С�뾶
//min_Radius��Բ���뾶
void houghcircles(cv::Mat& src_gray, Mat edges, std::vector<cv::Vec3f>& circles,
    double min_dist, int param2, int minRadius = 0, int maxRadius = 0);