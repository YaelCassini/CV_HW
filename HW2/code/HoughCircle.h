#pragma once

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

#define PI 3.14159265358979
#define HOUGH_CIRCLE_RADIUS_MIN						10  //圆最小半径
#define HOUGH_CIRCLE_RADIUS_MIN_DIST				2  //同心圆两个半径最小差
#define HOUGH_CIRCLE_INTEGRITY_DEGREE				0.6  //用于判断圆周上的点是否足够多（是否能成圆）
#define HOUGH_CIRCLE_SAMEDIRECT_DEGREE				0.99  //用于梯度检测
#define HOUGH_CIRCLE_GRADIENT_INTEGRITY_DEGREE		0.9   //用于梯度检测

//记录圆心
struct Centers2
{
    int x;
    int y;
    int count;
};
//用于排序
bool center_order(Centers2 a, Centers2 b);

//记录其他点与圆心的距离和内积
struct Radius
{
    double dist2;
    double inner_product;
};
//用于排序
bool radius_order(Radius a, Radius b);
//用于归一化圆心检测图像并放大差异
float normalization(float x);

//hough梯度法检测圆形
//src_gray：灰度图像
//edges：边缘图像
//circles：储存结果圆参数
//min_dist：两圆心之间的最小距离，小于该距离两圆心将被合并
//add_threshold：阈值，累加器中大于该值的点才能被选定为圆心
//min_Radius：圆最小半径
//min_Radius：圆最大半径
void houghcircles(cv::Mat& src_gray, Mat edges, std::vector<cv::Vec3f>& circles,
    double min_dist, int param2, int minRadius = 0, int maxRadius = 0);