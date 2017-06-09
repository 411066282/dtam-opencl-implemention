#pragma once

#include "utils.hpp"
#include "RunCL.h"

typedef int FrameID;
#define CONSTT uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf

typedef size_t st;
#define downp (point+w)
#define upp (point-w)
#define rightp (point+1)
#define leftp (point-1)
#define here (point)
#define gdown gd[here]
#define gup gu[here]
#define gleft gl[here]
#define gright gr[here]

class CostVol
{
public:
	CostVol()
	{};
	~CostVol();

	FrameID fid;
	int rows;
	int cols;
	int layers;
	float near; //inverse depth of center of voxels in layer layers-1
	float far;  //inverse depth of center of voxels in layer 0
	float depthStep;
	float initialWeight;
	cv::Mat R;
	cv::Mat T;
	cv::Mat cameraMatrix;//Note! should be in OpenCV format
	float occlusionThreshold;
	
	RunCL cvrc;

	cv::Mat projection;//projects world coordinates (x,y,z) into (rows,cols,layers)
	cv::Mat baseImage, baseImageGray;
    //cv::Mat _qx, _qy, _d, _a, _g, _gu, _gd, _gl, _gr, _gbig;
	cv::Mat _qx, _qy, _d, _a, _g, _g1, _gx, _gy, lo, hi;
	Mat costdata, hit;
	int count, QDruncount, Aruncount;
	void updateCost(const cv::Mat& image, const cv::Mat& R, const cv::Mat& T);//Accepts pinned RGBA8888 or BGRA8888 for high speed
	
	CostVol(cv::Mat image, FrameID _fid, int _layers, float _near, float _far,
		cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix, float occlusionThreshold,
		 float initialCost = 3.0, float initialWeight = .001);

	void initOptimization();

	//void optimizeQD();

	//bool optimizeA();

	void updateQD();

	bool updateA();

	void computeSigmas(float epsilon, float theta)
	{
		float lambda, alpha, gamma, delta, mu, rho, sigma;
		float L = 4;//lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44

		lambda = 1.0 / theta;
		alpha = epsilon;

		gamma = lambda;
		delta = alpha;

		mu = 2.0*std::sqrt(gamma*delta) / L;

		rho = mu / (2.0*gamma);
		sigma = mu / (2.0*delta);

		sigma_d = rho;
		sigma_q = sigma;
	}

	void cacheGValues();

	void GetResult();
private:
	cv::Mat cBuffer;//Must be pagable
	cv::Ptr<char> ref;

	void solveProjection(const cv::Mat& R, const cv::Mat& T);
	void checkInputs(const cv::Mat& R, const cv::Mat& T,
		const cv::Mat& _cameraMatrix);
	
	float theta, thetaStart, thetaStep, thetaMin, epsilon, lambda, sigma_d, sigma_q;
	float alloced,cachedG,dInited;
	//void cacheGValues();
};

