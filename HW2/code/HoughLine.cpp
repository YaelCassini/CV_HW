#include "HoughLine.h"
//用于排序
bool polar_order(Polar a, Polar b)
{
    return (a.count > b.count);
}

void houghlines(Mat img, vector<Line>& lines, int threshold, double rho, double theta)
{
    //图像长宽
    int w = img.cols;
    int h = img.rows;
    
    //累加器大小
    int add_w = 180/theta;
    int add_h = 1.5 * (w + h) /rho ;
    //消除值为负的ρ
    int center_h = add_h / 2;


    //为累加器分配空间
    int** add = (int**)malloc(add_h * sizeof(int*));
    for (int i = 0; i < add_h; i++)
        add[i] = (int*)malloc((add_w + 1) * sizeof(int));

    //累加器赋值为0
    for (int i = 0; i < add_h; i++)
        for (int j = 0; j < add_w; j++)
            add[i][j] = 0;


    //累加器投票
    int threshold_pix = 200;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            if ((int)img.at<uchar>(i, j) > threshold_pix)
            {
                for (int k = 0; k < 180; k=k+theta) {
                    double angle = (double)k / 180 * PI;  //角度制转换为弧度值
                    double dr = (double)j * cos(angle) + (double)i * sin(angle);
                    int r = round(dr)/rho;
                    int kk = k/theta;
                    add[r + center_h][kk]++;  //r可为负值，加上矩阵中心                  
                }
            }
        }
    }

    
    //遍历累加器，选出符合规定的ρ-θ点对作为检测到的直线参数
    vector<Polar> v;  
    for (int y = 1; y < add_h - 1; y++)//模长
    {
        for (int x = 0; x < add_w; x++)//角度
        {
            int flag = 0;
            //如果当前点在累加器边界处
            if (x == 0)
            {
                if (add[y][x] > threshold && add[y][x] > add[y][x + 1] && add[y][x] > add[y - 1][x] && add[y][x] > add[y + 1][x]) 
                    flag = 1;
            }
            else if (x == add_w - 1)
            {
                if (add[y][x] > threshold && add[y][x] > add[y][x - 1] && add[y][x] > add[y - 1][x] && add[y][x] > add[y + 1][x])
                    flag = 1;
            }
            //如果当前的值大于阈值，并在4邻域内它是最大值，则该点被认为是圆心
            else if (add[y][x] > threshold && add[y][x] > add[y][x - 1] &&
                add[y][x] > add[y][x + 1] && add[y][x] > add[y - 1][x] &&
                add[y][x] > add[y + 1][x])
            {
                flag = 1;
            }
            //选择该点表示的参数作为一条直线的参数
            if (flag)
            {
                Polar po;
                po.x = y;    //ρ
                po.y = x;    //θ
                po.count = add[y][x];
                //聚合，如果当前参数对被选中，则把相邻(-5,5)区间内的累加器值都加给它
                for (int p = y - 5; p <= y + 5; p++)
                {
                    for (int q = x - 5; q <= x + 5; q++)
                    {
                        if (p >= 0 && p <= add_h - 1 && q >= 0 && q <= add_w - 1)
                            po.count += add[p][q];
                    }
                }
                v.push_back(po); //当前点符合要求，作为圆心
            }

        }
    }
    /*
    for (int i = 0; i < add_h; i++) {  //找出投票数局部最大的r， theta组合。
        for (int j = 0; j < add_w; j++) {
            if (add[i][j] > threshold) {*/
                /*
                Position po;
                po.x = i;    //r
                po.y = j;    //theta
                po.count = add[i][j];
                v.push_back(po);  //将局部最大值的点存进一个vector中
                */ 
    /*
                if (i == 220 && j == 45)
                    int mm = 1;
                int flag = 1;
                if (i > 0) {
                    if (j > 0) {
                        if (add[i][j] < add[i - 1][j - 1]) flag = 0;
                        if (add[i][j] < add[i][j - 1]) flag = 0;
                    }
                    if (j < add_w - 1) {
                        if (add[i][j] < add[i - 1][j + 1]) flag = 0;
                        if (add[i][j] < add[i][j + 1]) flag = 0;
                    }
                    if (add[i][j] < add[i - 1][j]) flag = 0;
                }
                if (i < add_h - 1) {
                    if (j > 0) {
                        if (add[i][j] < add[i + 1][j - 1]) flag = 0;
                    }
                    if (j < add_w - 1) {
                        if (add[i][j] < add[i + 1][j + 1]) flag = 0;
                    }
                    if (add[i][j] < add[i + 1][j]) flag = 0;
                }

                if (flag == 1) {

                    Polar po;
                    po.x = i;    //r
                    po.y = j;    //theta
                    po.count = add[i][j];
                    for (int p = i - 5; p <= i + 5; p++)
                    {
                        for (int q = j - 5; q <= j + 5; q++)
                        {
                            if (p >= 0 && p <= add_h - 1 && q >= 0 && q <= add_w - 1)po.count += add[p][q];
                        }
                    }
                    v.push_back(po);  //将局部最大值的点存进一个vector中
                }
            }
        }
    }*/
  
    //对检测到的参数对按照累加器中的值大小降序排列
    sort(v.begin(), v.end(), polar_order);
    //将投票数最多的直线转换到笛卡尔坐标系计算出起始点坐标并存入名为lines的vector
    vector<Polar>::iterator iter;
    for (iter = v.begin(); iter != v.end(); iter++) {
        int x1, y1, x2, y2;
        x1 = y1 = x2 = y2 = 0;
        double angle = (double)(iter->y) / 180 * PI;
        double si = sin(angle);
        double co = cos(angle);

        if (iter->y == 0)
        {
            y1 = 0;
            y2 = h - 1;
            x1 = x2 = get_position(img, iter->x, iter->y, false);
            Line temp;
            temp.start = Point(x1, y1);
            temp.end = Point(x2, y2);
            lines.push_back(temp);
        }
        else if (iter->y == 90)
        {
            x1 = 0;
            x2 = w - 1;
            y1 = y2 = get_position(img, iter->x, iter->y, true);
            Line temp;
            temp.start = Point(x1, y1);
            temp.end = Point(x2, y2);
            lines.push_back(temp);
        }
        else if (iter->y >= 45 && iter->y <= 135) {//在这个范围内sin值比较大，使用sin做分母误差较小
            x1 = 0;
            y1 = (iter->x - center_h) / si; //加上之前为了消除ρ的负值减去的值
            x2 = w - 1;
            y2 = ((iter->x - center_h) - (double)x2 * co) / si;
            Line temp;
            temp.start = Point(x1, y1);
            temp.end = Point(x2, y2);
            lines.push_back(temp);
        }
        else { //在这个范围内cos值比较大，使用cos做分母误差较小      
            y1 = 0;
            x1 = (iter->x - center_h) / co;  //加上之前为了消除ρ的负值减去的值
            y2 = h - 1;
            x2 = ((iter->x - center_h) - (double)y2 * si) / co;
            Line temp;
            temp.start = Point(x1, y1);
            temp.end = Point(x2, y2);
            lines.push_back(temp);
        }
    }

}

//对于水平和垂直的线条找回丢失的坐标信息,参数flag用于表示是水平直线还是垂直直线
int get_position(Mat img, int ii, int jj,int flag ,int rho, int theta)
{
    int w = img.cols;
    int h = img.rows;
    int add_w = 180 /theta;
    int add_h = 1.5 * (w + h) /rho;
    int center_h = add_h / 2;
    //vector<int> vec_i;
    //vector<int> vec_j;
    int x;
    int y;

    int threshold_pix = 200;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            if ((int)img.at<uchar>(i, j) > threshold_pix)
            {
                for (int k = 0; k < 180; k = k + theta) {
                    double angle = (double)k / 180 * PI;  //角度制转换为弧度值
                    double dr = (double)j * cos(angle) + (double)i * sin(angle);
                    int r = round(dr);
                    if (r + center_h == ii && k/theta == jj)
                    {
                        if (flag)
                        {
                            x = i;                 
                        }
                        else
                        {
                            y = j;
                        }
                    }
                }
            }
        }
    }
    
    if (flag)return x;
    else return y;
}

