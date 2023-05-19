#include "HoughLine.h"
//��������
bool polar_order(Polar a, Polar b)
{
    return (a.count > b.count);
}

void houghlines(Mat img, vector<Line>& lines, int threshold, double rho, double theta)
{
    //ͼ�񳤿�
    int w = img.cols;
    int h = img.rows;
    
    //�ۼ�����С
    int add_w = 180/theta;
    int add_h = 1.5 * (w + h) /rho ;
    //����ֵΪ���Ħ�
    int center_h = add_h / 2;


    //Ϊ�ۼ�������ռ�
    int** add = (int**)malloc(add_h * sizeof(int*));
    for (int i = 0; i < add_h; i++)
        add[i] = (int*)malloc((add_w + 1) * sizeof(int));

    //�ۼ�����ֵΪ0
    for (int i = 0; i < add_h; i++)
        for (int j = 0; j < add_w; j++)
            add[i][j] = 0;


    //�ۼ���ͶƱ
    int threshold_pix = 200;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            if ((int)img.at<uchar>(i, j) > threshold_pix)
            {
                for (int k = 0; k < 180; k=k+theta) {
                    double angle = (double)k / 180 * PI;  //�Ƕ���ת��Ϊ����ֵ
                    double dr = (double)j * cos(angle) + (double)i * sin(angle);
                    int r = round(dr)/rho;
                    int kk = k/theta;
                    add[r + center_h][kk]++;  //r��Ϊ��ֵ�����Ͼ�������                  
                }
            }
        }
    }

    
    //�����ۼ�����ѡ�����Ϲ涨�Ħ�-�ȵ����Ϊ��⵽��ֱ�߲���
    vector<Polar> v;  
    for (int y = 1; y < add_h - 1; y++)//ģ��
    {
        for (int x = 0; x < add_w; x++)//�Ƕ�
        {
            int flag = 0;
            //�����ǰ�����ۼ����߽紦
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
            //�����ǰ��ֵ������ֵ������4�������������ֵ����õ㱻��Ϊ��Բ��
            else if (add[y][x] > threshold && add[y][x] > add[y][x - 1] &&
                add[y][x] > add[y][x + 1] && add[y][x] > add[y - 1][x] &&
                add[y][x] > add[y + 1][x])
            {
                flag = 1;
            }
            //ѡ��õ��ʾ�Ĳ�����Ϊһ��ֱ�ߵĲ���
            if (flag)
            {
                Polar po;
                po.x = y;    //��
                po.y = x;    //��
                po.count = add[y][x];
                //�ۺϣ������ǰ�����Ա�ѡ�У��������(-5,5)�����ڵ��ۼ���ֵ���Ӹ���
                for (int p = y - 5; p <= y + 5; p++)
                {
                    for (int q = x - 5; q <= x + 5; q++)
                    {
                        if (p >= 0 && p <= add_h - 1 && q >= 0 && q <= add_w - 1)
                            po.count += add[p][q];
                    }
                }
                v.push_back(po); //��ǰ�����Ҫ����ΪԲ��
            }

        }
    }
    /*
    for (int i = 0; i < add_h; i++) {  //�ҳ�ͶƱ���ֲ�����r�� theta��ϡ�
        for (int j = 0; j < add_w; j++) {
            if (add[i][j] > threshold) {*/
                /*
                Position po;
                po.x = i;    //r
                po.y = j;    //theta
                po.count = add[i][j];
                v.push_back(po);  //���ֲ����ֵ�ĵ���һ��vector��
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
                    v.push_back(po);  //���ֲ����ֵ�ĵ���һ��vector��
                }
            }
        }
    }*/
  
    //�Լ�⵽�Ĳ����԰����ۼ����е�ֵ��С��������
    sort(v.begin(), v.end(), polar_order);
    //��ͶƱ������ֱ��ת�����ѿ�������ϵ�������ʼ�����겢������Ϊlines��vector
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
        else if (iter->y >= 45 && iter->y <= 135) {//�������Χ��sinֵ�Ƚϴ�ʹ��sin����ĸ����С
            x1 = 0;
            y1 = (iter->x - center_h) / si; //����֮ǰΪ�������ѵĸ�ֵ��ȥ��ֵ
            x2 = w - 1;
            y2 = ((iter->x - center_h) - (double)x2 * co) / si;
            Line temp;
            temp.start = Point(x1, y1);
            temp.end = Point(x2, y2);
            lines.push_back(temp);
        }
        else { //�������Χ��cosֵ�Ƚϴ�ʹ��cos����ĸ����С      
            y1 = 0;
            x1 = (iter->x - center_h) / co;  //����֮ǰΪ�������ѵĸ�ֵ��ȥ��ֵ
            y2 = h - 1;
            x2 = ((iter->x - center_h) - (double)y2 * si) / co;
            Line temp;
            temp.start = Point(x1, y1);
            temp.end = Point(x2, y2);
            lines.push_back(temp);
        }
    }

}

//����ˮƽ�ʹ�ֱ�������һض�ʧ��������Ϣ,����flag���ڱ�ʾ��ˮƽֱ�߻��Ǵ�ֱֱ��
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
                    double angle = (double)k / 180 * PI;  //�Ƕ���ת��Ϊ����ֵ
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

