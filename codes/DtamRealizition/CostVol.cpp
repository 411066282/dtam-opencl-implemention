#include "stdafx.h"
#include "CostVol.h"
//#include <opencv2/core/operations.hpp>
#include <fstream>

using namespace std;
using namespace cv;

#define FLATALLOC(n) n.create(rows, cols, CV_32FC1); n.reshape(0, rows);

CostVol::~CostVol()
{
}

void CostVol::solveProjection(const cv::Mat& R, const cv::Mat& T) {
	Mat P;
	RTToP(R, T, P);
	//P:4x4 rigid transformation taking points from world to the camera frame
	//Camera:
	//fx 0  cx 
	//0  fy cy 
	//0  0  1  
	projection.create(4, 4, CV_64FC1);
	projection = 0.0;
	projection(Range(0, 2), Range(0, 3)) += cameraMatrix.rowRange(0, 2);
	//Projection:
	//fx 0  cx 0
	//0  fy cy 0
	//0  0  0  0
	//0  0  0  0

	projection.at<double>(2, 3) = 1.0;
	projection.at<double>(3, 2) = 1.0;

	//Projection: Takes camera coordinates to pixel coordinates:x_px,y_px,1/zc
	//fx 0  cx 0
	//0  fy cy 0
	//0  0  0  1
	//0  0  1  0

	Mat originShift = (Mat)(Mat_<double>(4, 4) << 1.0, 0., 0., 0.,
		0., 1.0, 0., 0.,
		0., 0., 1.0, -far,
		0., 0., 0., 1.0);

	projection = originShift*projection;//put the origin at 1/z_from_camera_center = far
	projection.row(2) /= depthStep;//stretch inverse depth so now x_cam,y_cam,z_cam-->x_cv_px, y_cv_px , [1/z_from_camera_center - far]_px
	projection = projection*P;//projection now goes x_world,y_world,z_world -->x_cv_px, y_cv_px , [1/z_from_camera_center - far]_px

							  // exit(0);
}

void CostVol::checkInputs(const cv::Mat& R, const cv::Mat& T,
	const cv::Mat& _cameraMatrix) {
	assert(R.size() == Size(3, 3));
	assert(R.type() == CV_64FC1);
	assert(T.size() == Size(1, 3));
	assert(T.type() == CV_64FC1);
	assert(_cameraMatrix.size() == Size(3, 3));
	assert(_cameraMatrix.type() == CV_64FC1);
	CV_Assert(_cameraMatrix.at<double>(2, 0) == 0.0);
	CV_Assert(_cameraMatrix.at<double>(2, 1) == 0.0);
	CV_Assert(_cameraMatrix.at<double>(2, 2) == 1.0);
}

CostVol::CostVol(Mat image, FrameID _fid, int _layers, float _near,
	float _far, cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix, float occlusionThreshold,
	float initialCost, float initialWeight)
	:
	R(R), T(T), occlusionThreshold(occlusionThreshold), initialWeight(initialWeight)
{

	//For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
	CV_Assert(image.rows % 32 == 0 && image.cols % 32 == 0 && image.cols >= 64);
	//     CV_Assert(_layers>=8);

	checkInputs(R, T, _cameraMatrix);
	fid = _fid;
	rows = image.rows;
	cols = image.cols;
	layers = _layers;
	near = _near;
	far = _far;
	depthStep = (near - far) / (layers - 1);
	cameraMatrix = _cameraMatrix.clone();
	solveProjection(R, T);
	costdata = Mat::zeros(layers, rows * cols, CV_32FC1);
	costdata = initialCost;
	hit = Mat::zeros(layers, rows * cols, CV_32FC1);
	hit = initialWeight;
	
	FLATALLOC(lo);
	FLATALLOC(hi);
	FLATALLOC(_a);
	FLATALLOC(_d);
	FLATALLOC(_gx);
	FLATALLOC(_gy);
	FLATALLOC(_qx);
    FLATALLOC(_qy);
    FLATALLOC(_g1);
	_qx = _qy  = 0;
	_gx = _gy = 1;
	cvrc.width = cols;
	cvrc.height = rows;
	cvrc.allocatemem((float*)_qx.data, (float*)_qy.data, (float*)_gx.data, (float*)_gy.data);

	image.copyTo(baseImage);
	baseImage = baseImage.reshape(0, rows);
	cvtColor(baseImage, baseImageGray, CV_RGB2GRAY);
	baseImageGray = baseImageGray.reshape(0, rows);
	count = 0;
	
	float off = layers / 32;
	thetaStart = 200.0*off;
	thetaMin = 1.0*off;
	thetaStep = .97;
	epsilon = .1*off;
	lambda = .001 / off;
	theta = thetaStart;

	QDruncount = 0;
	Aruncount = 0;

	alloced = 0;
	cachedG = 0;
	dInited = 0;
}

void CostVol::updateCost(const Mat& _image, const cv::Mat& R, const cv::Mat& T) 
{
	Mat image;
	_image.copyTo(image);
	//find projection matrix from cost volume to image (3x4)
	Mat viewMatrixImage;
	RTToP(R, T, viewMatrixImage);
	Mat cameraMatrixTex(3, 4, CV_64FC1);
	cameraMatrixTex = 0.0;
	cameraMatrix.copyTo(cameraMatrixTex(Range(0, 3), Range(0, 3)));
	cameraMatrixTex(Range(0, 2), Range(2, 3)) += 0.5;//add 0.5 to x,y out //removing causes crash
	Mat imFromWorld = cameraMatrixTex*viewMatrixImage;//3x4
	Mat imFromCV = imFromWorld*projection.inv();
	assert(baseImage.isContinuous());
	assert(lo.isContinuous());
	assert(hi.isContinuous());
	
	double *p = (double*)imFromCV.data;
	float persp[12];
	for (int i = 0; i<12; i++) persp[i] = p[i];
	image = image.reshape(0, rows);
	
	//memcpy(costd, (float*)costdata.data, st);
	//float* hitd = (float*)malloc(st);
	//memcpy(hitd, (float*)hit.data, st);
	
	cvrc.calcCostVol(persp, baseImage, image, (float*)costdata.data, (float*)hit.data, occlusionThreshold, layers);
	//memcpy(hit.data, hitd, st);
	/*size_t st = rows * cols * layers * sizeof(float);
	float* costd = (float*)malloc(st);
	memcpy(costd ,costdata.data, st);

	double min = 0, max = 0;
	minMaxIdx(costdata, &min, &max);*/
}

void CostVol::cacheGValues()
{
	cvrc.cacheGValue(baseImageGray);
}

void CostVol::updateQD()
{
	computeSigmas(epsilon, theta);

	cvrc.updateQD(epsilon, theta, sigma_q, sigma_d);
}

bool CostVol::updateA()
{
	bool doneOptimizing = theta <= thetaMin;

	cvrc.updateA(layers,lambda,theta);
	
	theta *= thetaStep;

	return doneOptimizing;
}

void CostVol::GetResult()
{
	cvrc.ReadOutput((float*)_a.data);
	cvrc.CleanUp();
}

