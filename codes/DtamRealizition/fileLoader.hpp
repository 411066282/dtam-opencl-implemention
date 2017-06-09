#ifndef FILE_LOADER
#define FILE_LOADER
#include "opencv\cv.h"
// loads all files of a given name and extension
#include "convertAhandaPovRayToStandard.h"
void loadAhanda(const char * rootpath,
                double range,
                int imageNumber,
                cv::Mat& image,
                cv::Mat& d,
                cv::Mat& cameraMatrix,
                cv::Mat& R,
                cv::Mat& T);
cv::Mat loadDepthAhanda(std::string filename, int r,int c,cv::Mat cameraMatrix);

#endif