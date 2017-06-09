// DtamRealizition.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include <boost/filesystem.hpp>
#include "CostVol.h"

#include "fileLoader.hpp"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;
int main()
{
	//initialize opencl
	int numImg = 50;
	char filename[500];
	std::string pathDepthDir = "D:\\projects\\";
	Mat image, R, T;
	Mat cameraMatrix = (Mat_<double>(3, 3) << 481.20, 0.0, 319.5,
		0.0, 480.0, 239.5,
		0.0, 0.0, 1.0);

	vector<Mat> images, Rs, ds, Ts, Rs0, Ts0, D0;
	double reconstructionScale = 1;
	int inc = 1;
	for (int i =0; i < 11; i += inc) {
		Mat tmp, d, image;
		int offset = 0;
		loadAhanda("D:\\projects\\DTAM-master\\DTAM-master\\Trajectory_30_seconds\\",
			65535,
			i + offset,
			image,
			d,
			cameraMatrix,
			R,
			T);
		images.push_back(image);
		Rs.push_back(R.clone());
		Ts.push_back(T.clone());
		ds.push_back(d.clone());
		Rs0.push_back(R.clone());
		Ts0.push_back(T.clone());
		D0.push_back(1 / d);
	}
	//Setup camera matrix
	double sx = reconstructionScale;
	double sy = reconstructionScale;

	int layers = 32;
	int imagesPerCV = 10;
	float occlusionThreshold = .05;

	int startAt = 0;

	CostVol cv(images[startAt], (FrameID)startAt, layers, 0.015, 0.0, Rs[startAt], Ts[startAt], cameraMatrix, occlusionThreshold);
	cout << "calculate cost volume: " << endl;

	for (int imageNum = 1; imageNum < 10; imageNum++)
		cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);

	cout << "cacheGValues: " << endl;
	cv.cacheGValues();

	cout << "optimizing: " << endl;
	bool doneOptimizing;

	do
	{
		for (int i = 0; i < 10; i++)
			cv.updateQD();
		doneOptimizing = cv.updateA();
	} while (!doneOptimizing);

	cout << "done: " << endl;
	
	cv::Mat depthMap;
	double min = 0, max = 0;
	cv::Mat depthImg;
	cv::namedWindow("depthmap", 1);

	cv.GetResult();
    depthMap = cv._a;

	cv::minMaxIdx(depthMap, &min, &max);
	std::cout << "Max: " << max << " Min: " << min << std::endl;
	depthMap.convertTo(depthImg, CV_8U, 255 / max, 0);
	medianBlur(depthImg, depthImg, 3);
	cv::imshow("depthmap", depthImg);
	cv::waitKey(-1);
	return 0;
}

