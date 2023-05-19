// Author
#include "HoughCircle.h"

void houghcircles(cv::Mat& src_gray, Mat edges, std::vector<cv::Vec3f>& circles, double min_dist,
	int add_threshold, int minRadius, int maxRadius) 
{
	int rows = src_gray.rows;    //?????
	int cols = src_gray.cols;    //?????		

	//?????��???????��?	
	if (minRadius < HOUGH_CIRCLE_RADIUS_MIN)
		minRadius = HOUGH_CIRCLE_RADIUS_MIN;
	if (maxRadius < minRadius * 2)
		maxRadius = MAX(rows, cols);

	int i, j, k, center_count, points_count;
	//??��?????????????
	float minRadius2 = (float)minRadius * minRadius;
	float maxRadius2 = (float)maxRadius * maxRadius;

	//?????????????????????????��????
	const int SHIFT = 10, ONE = 1 << SHIFT;
	//?????????????????????
	Mat dx, dy;
	Mat dxdy;
	dxdy = cv::Mat(src_gray.rows, src_gray.cols, CV_32SC2);


	//???��?????????
	vector<cv::Point> points;
	//????????????
	vector<Centers2> centers2;
	//???????
	vector<Radius> radius;
	//??????????
	double dp = 1.0;
	//????
	float idp = 1.f / dp;
	//???????
	float dr = dp;

	//???????????????????��
	int add_rows = (double)src_gray.rows / dp + 0.5;
	int add_cols = (double)src_gray.cols / dp + 0.5;


	//??????????????????????????
	int** add = (int**)malloc(add_rows * sizeof(int*));
	for (int i = 0; i < add_rows; i++)
	{
		add[i] = (int*)malloc((add_cols) * sizeof(int));
	}
	//?????????0
	for (int i = 0; i < add_rows; i++) {
		for (int j = 0; j < add_cols; j++) {
			add[i][j] = 0;
		}
	}

	//???Sobel??????????????????
	cv::Sobel(src_gray, dx, CV_16SC1, 1, 0, 3);
	cv::Sobel(src_gray, dy, CV_16SC1, 0, 1, 3);

	//?????????��?????
 	int accum_max = 0;
	//??????????
	for (int y = 0; y < rows; y++)
	{
		//???????????????????????????????��?????
		const uchar* edges_row = edges.ptr<uchar>(y);
		const short* dx_row = (const short*)dx.ptr<short>(y);
		const short* dy_row = (const short*)dy.ptr<short>(y);
		cv::Vec2i* dxdy_row = (cv::Vec2i*)dxdy.ptr<cv::Vec2i>(y);

		for (int x = 0; x < cols; x++)
		{
			float dx_now, dy_now;
			int sx, sy, x1, y1, r;
			cv::Point pt;
			//???????????????????
			dx_now = dx_row[x];
			dy_now = dy_row[x];
			//????????????????????????????????????????0
			if (!edges_row[x] || (dx_now == 0 && dy_now == 0))
				continue;
			//??????????????
			float length = sqrt(dx_now * dx_now + dy_now * dy_now);
			if (length < 1)
			{
				cout << "Error!" << endl;
				return;
			}

			//??��??????????????????????��??
			sx = cvRound((dx_now *idp) / length * ONE );
			sy = cvRound((dy_now *idp) / length * ONE );

			//???�Ŧ�???????????????????????
			dxdy_row[x][0] = sx;
			dxdy_row[x][1] = sy;

			//cout << sx << " " << sy << endl;

			//???????????????????????????????
			int x_now = cvRound((x *idp) * ONE);
			int y_now = cvRound((y *idp) * ONE);
			

			//?????????????????????????????????
			for (int times = 0; times < 2; times++)
			{
				//?????????��???????��??
				x1 = x_now + minRadius * sx;
				y1 = y_now + minRadius * sy;

				//???????????��???
				for (r = minRadius; r <= maxRadius; x1 += sx, y1 += sy, r++)
				{
					//?????????????????��
					int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
					//???��????????????????????��???????
					if ((unsigned)x2 >= (unsigned)add_cols ||(unsigned)y2 >= (unsigned)add_rows)
						break;
					//????????????��?????1
					add[y2][x2]++;
					//???????????????
					if (add[y2][x2] > accum_max) accum_max = add[y2][x2];
				}
				//??��???????????????
				sx = -sx; sy = -sy;
			}
			//??????????��?????
			pt.x = x; pt.y = y;
			points.push_back(pt);
		}
	}
	//???????��?????????????
	points_count = points.size();
	//????????0???????��???????????��???
	if (!points_count) return;


	//?????????????????????
	for (int y = 1; y < add_rows - 1; y++)
		for (int x = 1; x < add_cols - 1; x++)
			add[y][x] = normalization((float)add[y][x] / accum_max) * 256;


	//?????????
	for (int y = 1; y < add_rows - 1; y++)
	{
		for (int x = 1; x < add_cols - 1; x++)
		{
			//??????????????????????
			/*if (add[y][x] > add_threshold &&
				add[y][x] > add[y][x - 1] && 
				add[y][x] > add[y][x + 1] &&
				add[y][x] > add[y - 1][x] && 
				add[y][x] > add[y + 1][x] &&
				add[y][x] > add[y - 1][x-1] &&
				add[y][x] > add[y - 1][x+1] &&
				add[y][x] > add[y + 1][x-1] &&
				add[y][x] > add[y + 1][x+1])*/
			if (add[y][x] > add_threshold &&
					add[y][x] > add[y][x - 1] &&
					add[y][x] > add[y][x + 1] &&
					add[y][x] > add[y - 1][x] &&
					add[y][x] > add[y + 1][x])
			{
				//?????????????
				Centers2 temp;
				temp.x = x;
				temp.y = y;
				temp.count = add[y][x];
				centers2.push_back(temp);
			}
		}
	}

	//????????????
	Mat center_img = Mat(add_rows, add_cols, CV_32SC1);
	for (int i = 0; i < rows; i++)
	{
		int* ptmp = center_img.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			center_img.at<int>(i, j) = add[i][j];
		}
	}
	cv::normalize(center_img, center_img, 0, 255, cv::NORM_MINMAX);
	center_img.convertTo(center_img, CV_8UC1);
	imshow("centers", center_img);
	waitKey(0);

	//?????????????????
	center_count = centers2.size();
	if (!center_count) return;

	//????????��?????????��????????????
	sort(centers2.begin(), centers2.end(), center_order);


	//????????????????��????
	min_dist = MAX(min_dist, dp);
	//??��????????
	min_dist *= min_dist;

	//??????????��?????��???????????????????
	for (i = 0; i < centers2.size(); i++)
	{
		radius.clear();

		int y = centers2[i].y;
		int x = centers2[i].x;

		//?????????????????��?????��??
		float cx = (float)((x + 0.5f) * dp), cy = (float)((y + 0.5f) * dp);
		float start_dist, dist_sum;

		//?��?????????????????????
		for (j = 0; j < circles.size(); j++)
		{
			cv::Vec3f center = circles[j];
			if ((center[0] - cx) * (center[0] - cx) + (center[1] - cy) * (center[1] - cy) < min_dist)
				break;
		}
		if (j < circles.size())
			continue;

		//?????????????????��????��?
		for (j = k = 0; j < points_count; j++)
		{
			cv::Point pt;
			pt = points[j];

			//??????????��????????????
			float _dx, _dy, _r2;
			_dx = cx - pt.x; _dy = cy - pt.y;
			_r2 = _dx * _dx + _dy * _dy;

			float x_norm = _dx / pow(_r2, 0.5);
			float y_norm = _dy / pow(_r2, 0.5);

			cv::Vec2i dxdy_row = dxdy.at<cv::Vec2i>(pt.y, pt.x);
			short sx = dxdy_row[0];
			short sy = dxdy_row[1];
			
			//cout << x_norm << " " << y_norm << endl;
			//cout << sx << " " << sy << endl;

			/*
			//??��??????????????????????��??
			sx = cvRound((dx_now * idp) / mag * ONE);
			sy = cvRound((dy_now * idp) / mag * ONE);*/

			//??????????��??????
			if (minRadius2 <= _r2 && _r2 <= maxRadius2)
			{
				k++;
				Radius temp;
				temp.dist2 = _r2;
				temp.inner_product = sx * x_norm + sy * y_norm;
				radius.push_back(temp);
			}
		}
		//k???????��???????????
		int point_cnt1 = k, start_idx = point_cnt1 - 1;
		if (point_cnt1 == 0)
			continue;

		//?????????????????????
		for (int t = 0; t < point_cnt1; ++t) {
			//ddata[t] = pow(ddata[t], 0.5);
			radius[t].dist2 = pow(radius[t].dist2, 0.5);

		}

		//????????????????????��??????????
		sort(radius.begin(), radius.end(), radius_order);

		start_dist = radius[0].dist2;
		float cur_r_dist_sum = 0.0;
		int cur_r_count = 0;
		int cur_r_grad_count = 0;


		//???????????????????????????
		for (j = 0; j <point_cnt1; j++)
		{

			float dist2 = radius[j].dist2;
			float inner_product = radius[j].inner_product;
		
			if (dist2 > maxRadius) break;
			if (dist2 - start_dist < HOUGH_CIRCLE_RADIUS_MIN_DIST * dr)
			{
				//??????????
				cur_r_count++;
				cur_r_dist_sum += dist2;
				//?????
				if (fabs(inner_product) > 0.99 * ONE) {
					cur_r_grad_count++;
				}
			}
			//??????????????????��???????
			else {
				//?????????
				float r_mean = cur_r_dist_sum / cur_r_count;	
				if (cur_r_count >= HOUGH_CIRCLE_INTEGRITY_DEGREE * 2 * PI * r_mean &&
					cur_r_grad_count >= 0.9 * cur_r_count) 
				{
					cv::Vec3f c;
					c[0] = cx;    //?????????
					c[1] = cy;    //??????????
					c[2] = (float)r_mean;    //???????????
					circles.push_back(c);    //???????circles??				
				}
				cur_r_count = 1;
				cur_r_dist_sum = dist2;
				start_dist = dist2;
				
			}
		}
		//???????????????
		//?????????
		float r_mean = cur_r_dist_sum / cur_r_count;
		// ?��????????
		if (cur_r_count >= HOUGH_CIRCLE_INTEGRITY_DEGREE * 2 * PI * r_mean &&
			cur_r_grad_count >= 0.9 * cur_r_count
			) 
		{
			cv::Vec3f c;
			c[0] = cx;    //?????????
			c[1] = cy;    //??????????
			c[2] = (float)r_mean;    //???????????
			circles.push_back(c);    //???????circles??
		}
	}
}


bool center_order(Centers2 a, Centers2 b)
{ return (a.count > b.count); }
bool radius_order(Radius a, Radius b) 
{ return (a.dist2 < b.dist2); }


//????????��???????????????
float normalization(float x) {
	float result = 0.0;
	double r1 = 0.3;
	double r2 = 0.5;
	double s1 = 0.2;
	double s2 = 0.8;
	if (x >= 0 && x < r1) {
		result = s1 / r1 * x;
	}
	else if (x >= r1 && x < r2) {
		result = (s2 - s1) / (r2 -
			r1) * (x - r1) + s1;
	}
	else {
		result = (1 - s2) / (1 - r2) * (x - 1) + 1;
	}
	return result;
}