#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

#define PI 3.1415926

// ρ-θ空间累加器
struct Polar
{
    int x;
    int y;
    int count;
};
//用于返回直线两端的坐标
struct Line
{
    Point start;
    Point end;
};
bool polar_order(Polar a, Polar b);  //用于排序
int get_position(Mat img, int ii, int jj, int flag, int rho=1, int theta=1);  //对于水平和垂直的直线，找回Hough变换时损失的信息
//hough变换检测直线
//img：输入的边缘图像（灰度图，只有0,255两个值）
//lines：储存最后检测出的直线起始点
//threshold：累加器阈值，只有累加器中的值大于该值，该参数对才会被选中
//rho：ρ参数的分辨率（遍历步长）
//theta：θ参数的分辨率（遍历步长）
void houghlines(Mat img,vector<Line>& lines, int threshold, double rho = 1, double theta = 1);