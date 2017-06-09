#include "stdafx.h"
#include "RunCL.h"

#define SDK_SUCCESS 0
#define SDK_FAILURE 1
using namespace std;

RunCL::RunCL()
{
	/*Step1: Getting platforms and choose an available one.*/
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		cout << "Error: Getting platforms!" << endl;
		return;
	}

	/*For clarity, choose the first available platform. */
	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];
		free(platforms);
	}

	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}


	/*Step 3: Create context.*/
	 cl_context_properties cps[3] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		0
	};

	m_context = clCreateContextFromType(
										cps,
										CL_DEVICE_TYPE_GPU,
										NULL,
										NULL,
										&status);
	/*Step 4: Creating command queue associate with the context.*/
	deviceId = devices[0];
	cl_command_queue_properties prop[] = { 0 };
	m_queue = clCreateCommandQueueWithProperties(m_context,
													deviceId,
													prop,
													&status);

	/*Step 5: Create program object */
	const char *filename = "DTAM_kernels2.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	m_program = clCreateProgramWithSource(m_context, 1, &source, sourceSize, NULL);

	/*Step 6: Build program. */
	status = clBuildProgram(m_program, 1, devices, NULL, NULL, NULL);
	
	if (status != CL_SUCCESS)
	{
		printf("clBuildProgram failed: %d\n", status);
		char buf[0x10000];
		clGetProgramBuildInfo(m_program, deviceId, CL_PROGRAM_BUILD_LOG, 0x10000, buf, NULL);
		printf("\n%s\n", buf);
		return;
	}
	//*Step 7: Create kernel object. */
	cost_kernel = clCreateKernel(m_program, "BuildCostVolume", NULL);
	//min_kernel = clCreateKernel(m_program, "CostMin", NULL);
	//optiQ_kernel = clCreateKernel(m_program, "OptimizeQ", NULL);
	//optiD_kernel = clCreateKernel(m_program, "OptimizeD", NULL);
	//optiA_kernel = clCreateKernel(m_program, "OptimizeA", NULL);
	cache1_kernel = clCreateKernel(m_program, "CacheG1", NULL);
	cache2_kernel = clCreateKernel(m_program, "CacheG2", NULL);
	updateQ_kernel = clCreateKernel(m_program, "UpdateQ", NULL);
	updateD_kernel = clCreateKernel(m_program, "UpdateD", NULL);
	updateA_kernel = clCreateKernel(m_program, "UpdateA", NULL);
}

void RunCL::calcCostVol(float* p, cv::Mat &baseImage, cv::Mat &image, float *cdata, float *hdata, float thresh, int layers)
{
	cl_int status;
	cl_int res;
	size_t s = baseImage.total() * 3;
	cl_event writeEvt;
	int pixelSize = baseImage.channels();

	if (basemem == 0)
	{
		imgmem = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, s, image.ptr(), &res);
		pbuf = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 12 * sizeof(float), p, &res);
		basemem = clCreateBuffer(m_context, CL_MEM_READ_ONLY | (1 << 6), s, 0, &res);
		cdatabuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * layers * sizeof(float), 0, &res);
		hdatabuf = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * layers * sizeof(float), 0, &res);

		if (0 == basemem || CL_SUCCESS != res || 0 == imgmem)
			return;

		///////////////////create other data buffer
		status = clEnqueueWriteBuffer(m_queue,
			basemem,
			CL_FALSE,
			0,
			width * height * pixelSize,
			baseImage.data,
			0,
			NULL,
			&writeEvt);
		status = clEnqueueWriteBuffer(m_queue,
			imgmem,
			CL_FALSE,
			0,
			width * height * pixelSize,
			image.data,
			0,
			NULL,
			&writeEvt);

		status = clEnqueueWriteBuffer(m_queue,
			cdatabuf,
			CL_FALSE,
			0,
			width * height * layers * sizeof(float),
			cdata,
			0,
			NULL,
			&writeEvt);

		status = clEnqueueWriteBuffer(m_queue,
			hdatabuf,
			CL_FALSE,
			0,
			width * height * layers * sizeof(float),
			hdata,
			0,
			NULL,
			&writeEvt);

		status = clEnqueueWriteBuffer(m_queue,
			pbuf,
			CL_FALSE,
			0,
			12 * sizeof(float),
			p,
			0,
			NULL,
			&writeEvt);
		status = clFlush(m_queue);
		status = waitForEventAndRelease(&writeEvt);
		int layerstep = width * height;
		global_work_size = layerstep;
		// set kernelArg
		res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem), &pbuf);
		res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem), &basemem);
		res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem), &imgmem);
		res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem), &cdatabuf);
		res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem), &hdatabuf);
		res = clSetKernelArg(cost_kernel, 5, sizeof(int), &layerstep);
		res = clSetKernelArg(cost_kernel, 6, sizeof(float), &thresh);
		res = clSetKernelArg(cost_kernel, 7, sizeof(int), &width);
		res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem), &lomem);
		res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem), &himem);
		res = clSetKernelArg(cost_kernel, 10, sizeof(cl_mem), &amem);
		res = clSetKernelArg(cost_kernel, 11, sizeof(cl_mem), &dmem);
		res = clSetKernelArg(cost_kernel, 12, sizeof(int), &layers);

		cl_event ev;
		res = clEnqueueNDRangeKernel(m_queue, cost_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
	}
	else
	{
		status = clEnqueueWriteBuffer(m_queue,
			imgmem,
			CL_FALSE,
			0,
			width * height * pixelSize,
			image.data,
			0,
			NULL,
			&writeEvt);

		status = clEnqueueWriteBuffer(m_queue,
			pbuf,
			CL_FALSE,
			0,
			12 * sizeof(float),
			p,
			0,
			NULL,
			&writeEvt);
		status = clFlush(m_queue);
		status = waitForEventAndRelease(&writeEvt);
		int layerstep = width * height;
		global_work_size = layerstep;
		// set kernelArg
		res = clSetKernelArg(cost_kernel, 0, sizeof(cl_mem), &pbuf);
		res = clSetKernelArg(cost_kernel, 2, sizeof(cl_mem), &imgmem);
		res = clSetKernelArg(cost_kernel, 5, sizeof(int), &layerstep);
		res = clSetKernelArg(cost_kernel, 6, sizeof(float), &thresh);
		res = clSetKernelArg(cost_kernel, 7, sizeof(int), &width);
		res = clSetKernelArg(cost_kernel, 12, sizeof(int), &layers);
		cl_event ev;
		res = clEnqueueNDRangeKernel(m_queue, cost_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
	}
	// Enqueue read output buffer
	//cl_event readEvt;
	//status = clEnqueueReadBuffer(m_queue,
	//							cdatabuf,
	//							CL_FALSE,
	//							0,
	//							width * height * layers * sizeof(float),
	//							cdata,
	//							0,
	//							NULL,
	//							&readEvt);
	//status = clEnqueueReadBuffer(m_queue,
	//							hdatabuf,
	//							CL_FALSE,
	//							0,
	//							width * height * layers * sizeof(float),
	//							hdata,
	//							0,
	//							NULL,
	//							&readEvt);
//	status = clFlush(m_queue);
//	status = waitForEventAndRelease(&readEvt);
}

//void RunCL::minv(float *loInd, float *loVal, int layers)
//{
//	cl_int status;
//	cl_int res;
//	size_t si = height * width * sizeof(float);
//
//	cl_mem loi = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, si, 0, &res);
//	cl_mem lov = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, si, 0, &res);
//
//	if (0 == loi || CL_SUCCESS != res || 0 == lov)
//		return;
//		
//	int layerstep = width * height;
//	// set kernelArg
//	res = clSetKernelArg(min_kernel, 0, sizeof(cl_mem), &cdatabuf);
//	res = clSetKernelArg(min_kernel, 1, sizeof(cl_mem), &loi);
//	res = clSetKernelArg(min_kernel, 2, sizeof(cl_mem), &lov);
//	res = clSetKernelArg(min_kernel, 3, sizeof(int), &height);
//	res = clSetKernelArg(min_kernel, 4, sizeof(int), &layerstep);
//	cl_event ev;
//	res = clEnqueueNDRangeKernel(m_queue, min_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//	
//	// Enqueue read output buffer
//	cl_event readEvt;
//	status = clEnqueueReadBuffer(m_queue,
//		loi,
//		CL_FALSE,
//		0,
//		si,
//		loInd,
//		0,
//		NULL,
//		&readEvt);
//	
//	status = clFlush(m_queue);
//	status = waitForEventAndRelease(&readEvt);
//}
 
//void RunCL::optiQ(float sigma_q,float epsilon,float denom)
//{
//	cl_event ev;
//		
//	cl_int res;
//	res = clSetKernelArg(optiQ_kernel, 0, sizeof(float), &sigma_q);
//	res = clSetKernelArg(optiQ_kernel, 1, sizeof(float), &epsilon);
//	res = clSetKernelArg(optiQ_kernel, 2, sizeof(float), &denom);
//	res = clSetKernelArg(optiQ_kernel, 3, sizeof(int), &width);
//	res = clSetKernelArg(optiD_kernel, 4, sizeof(int), &height);
//	res = clSetKernelArg(optiQ_kernel, 5, sizeof(cl_mem), &qxmem);
//	res = clSetKernelArg(optiQ_kernel, 6, sizeof(cl_mem), &qymem);
//	res = clSetKernelArg(optiQ_kernel, 7, sizeof(cl_mem), &dmem);
//	res = clSetKernelArg(optiQ_kernel, 8, sizeof(cl_mem), &grmem);
//	res = clSetKernelArg(optiQ_kernel, 9, sizeof(cl_mem), &gdmem);
//	
//	res = clEnqueueNDRangeKernel(m_queue, optiQ_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//}
//
//void RunCL::optiD(float sigma_d, float epsilon, float denom, float theta)
//{
//	// set kernelArg
//	cl_event ev;
//	cl_int res = clSetKernelArg(optiD_kernel, 0, sizeof(float), &sigma_d);
//	res = clSetKernelArg(optiD_kernel, 1, sizeof(float), &epsilon);
//	res = clSetKernelArg(optiD_kernel, 2, sizeof(float), &denom);
//	res = clSetKernelArg(optiD_kernel, 3, sizeof(int), &width);
//	res = clSetKernelArg(optiD_kernel, 4, sizeof(int), &height);
//	res = clSetKernelArg(optiD_kernel, 5, sizeof(cl_mem), &qxmem);
//	res = clSetKernelArg(optiD_kernel, 6, sizeof(cl_mem), &qymem);
//	res = clSetKernelArg(optiD_kernel, 7, sizeof(cl_mem), &dmem);
//	res = clSetKernelArg(optiD_kernel, 8, sizeof(cl_mem), &gumem);
//	res = clSetKernelArg(optiD_kernel, 9, sizeof(cl_mem), &glmem);
//	res = clSetKernelArg(optiD_kernel, 10, sizeof(cl_mem), &grmem);
//	res = clSetKernelArg(optiD_kernel, 11, sizeof(cl_mem), &gdmem); 
//	res = clSetKernelArg(optiD_kernel, 12, sizeof(cl_mem), &amem);
//	res = clSetKernelArg(optiD_kernel, 13, sizeof(float), &theta);
//	
//	res = clEnqueueNDRangeKernel(m_queue, optiD_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//}
//
//void RunCL::optiA(float theta,float ds,float lamda,int l,int layerstep)
//{
//	cl_event ev;
//	cl_int res;
//	res = clSetKernelArg(optiA_kernel, 0, sizeof(cl_mem), &amem);
//	res = clSetKernelArg(optiA_kernel, 1, sizeof(cl_mem), &dmem);
//	res = clSetKernelArg(optiA_kernel, 2, sizeof(cl_mem), &cdatabuf);
//	res = clSetKernelArg(optiA_kernel, 3, sizeof(float), &theta);
//	res = clSetKernelArg(optiA_kernel, 4, sizeof(float), &ds);
//	res = clSetKernelArg(optiA_kernel, 5, sizeof(float), &lamda);
//	res = clSetKernelArg(optiA_kernel, 6, sizeof(int), &l);
//	res = clSetKernelArg(optiA_kernel, 7, sizeof(int), &layerstep);
//	res = clSetKernelArg(optiA_kernel, 8, sizeof(int), &width);
//
//
//	res = clEnqueueNDRangeKernel(m_queue, optiA_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
//}

void RunCL::allocatemem(float *qx,float *qy, float* gx, float* gy)
{
	cl_int res;
	qxmem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height  * sizeof(float), 0, &res);
	qymem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height  * sizeof(float), 0, &res);
	dmem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height  * sizeof(float), 0, &res);
	amem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	gxmem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	gymem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	gqxmem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	gqymem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	g1mem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	lomem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);
	himem = clCreateBuffer(m_context, CL_MEM_READ_WRITE | (1 << 6), width * height * sizeof(float), 0, &res);

	
	cl_int status;
	cl_event writeEvt;
	status = clEnqueueWriteBuffer(m_queue,
		qxmem,
		CL_FALSE,
		0,
		width * height,
		qx,
		0,
		NULL,
		&writeEvt);
	status = clEnqueueWriteBuffer(m_queue,
		qymem,
		CL_FALSE,
		0,
		width * height,
		qy,
		0,
		NULL,
		&writeEvt);
	
	
	status = clEnqueueWriteBuffer(m_queue,
		gxmem,
		CL_FALSE,
		0,
		width * height * sizeof(float),
		gx,
		0,
		NULL,
		&writeEvt);

	status = clEnqueueWriteBuffer(m_queue,
		gymem,
		CL_FALSE,
		0,
		width * height * sizeof(float),
		gy,
		0,
		NULL,
		&writeEvt);

	status = clFlush(m_queue);
	status = waitForEventAndRelease(&writeEvt);
}

RunCL::~RunCL()
{
	cl_int status;
	status = clReleaseKernel(cost_kernel);
	//status = clReleaseKernel(min_kernel);
	//status = clReleaseKernel(optiQ_kernel);
	//status = clReleaseKernel(optiD_kernel);
	//status = clReleaseKernel(optiA_kernel);
	status = clReleaseKernel(cache1_kernel);
	status = clReleaseKernel(cache2_kernel);
	status = clReleaseKernel(updateQ_kernel);
	status = clReleaseKernel(updateD_kernel);
	status = clReleaseKernel(updateA_kernel);

	status = clReleaseProgram(m_program);
	status = clReleaseCommandQueue(m_queue);

	status = clReleaseContext(m_context);
}

void RunCL::CleanUp()
{
	cl_int status;

	status = clReleaseMemObject(basemem);
	status = clReleaseMemObject(imgmem);
	status = clReleaseMemObject(cdatabuf);
	status = clReleaseMemObject(hdatabuf);
	status = clReleaseMemObject(pbuf);
	status = clReleaseMemObject(qxmem);
	status = clReleaseMemObject(qymem);
	status = clReleaseMemObject(gqxmem);
	status = clReleaseMemObject(gqxmem);
	status = clReleaseMemObject(dmem);
	status = clReleaseMemObject(amem);
	status = clReleaseMemObject(lomem);
	status = clReleaseMemObject(himem);
	/*status = clReleaseMemObject(gdmem);
	status = clReleaseMemObject(gumem);
	status = clReleaseMemObject(glmem);
	status = clReleaseMemObject(grmem);*/
}

void RunCL::cacheGValue(cv::Mat &bgray)
{
	cl_int status;
	cl_int res;
	size_t s = bgray.total();
	cl_event ev;

	if (basegraymem == 0)
	{
		basegraymem = clCreateBuffer(m_context, CL_MEM_READ_ONLY | (1 << 6), s, 0, &res);

		status = clEnqueueWriteBuffer(m_queue,
			basegraymem,
			CL_FALSE,
			0,
			width * height,
			bgray.data,
			0,
			NULL,
			&ev);
	}

	res = clSetKernelArg(cache1_kernel, 0, sizeof(cl_mem), &basegraymem);
	res = clSetKernelArg(cache1_kernel, 1, sizeof(cl_mem), &g1mem);
	res = clSetKernelArg(cache1_kernel, 2, sizeof(int), &width);
	res = clSetKernelArg(cache1_kernel, 3, sizeof(int), &height);
	res = clEnqueueNDRangeKernel(m_queue, cache1_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
	
	/*size_t st = width * height * sizeof(float);
	float* g1 = (float*)malloc(st);
	cl_event readEvt;
	status = clEnqueueReadBuffer(m_queue,
		g1mem,
		CL_FALSE,
		0,
		st,
		g1,
		0,
		NULL,
		&readEvt);
	status = clFlush(m_queue);
	status = waitForEventAndRelease(&readEvt);
	cv::Mat qx;
	qx.create(height, width, CV_32FC1); qx.reshape(0, height);
	memcpy(qx.data, g1, st);
	double min = 0, max = 0;
	cv::minMaxIdx(qx, &min, &max);*/

	res = clSetKernelArg(cache2_kernel, 0, sizeof(cl_mem), &g1mem);
	res = clSetKernelArg(cache2_kernel, 1, sizeof(cl_mem), &gxmem);
	res = clSetKernelArg(cache2_kernel, 2, sizeof(cl_mem), &gymem);
	res = clSetKernelArg(cache2_kernel, 3, sizeof(int), &width);
	res = clSetKernelArg(cache2_kernel, 4, sizeof(int), &height);
	res = clEnqueueNDRangeKernel(m_queue, cache2_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
}

void RunCL::updateQD(float epsilon, float theta, float sigma_q, float sigma_d)
{
	cl_int status;
	cl_int res;
	cl_event ev; 
	
	res = clSetKernelArg(updateQ_kernel, 0, sizeof(cl_mem), &gxmem);
	res = clSetKernelArg(updateQ_kernel, 1, sizeof(cl_mem), &gymem);
	res = clSetKernelArg(updateQ_kernel, 2, sizeof(cl_mem), &gqxmem);
	res = clSetKernelArg(updateQ_kernel, 3, sizeof(cl_mem), &gqymem);
	res = clSetKernelArg(updateQ_kernel, 4, sizeof(cl_mem), &dmem);
	res = clSetKernelArg(updateQ_kernel, 5, sizeof(float), &epsilon);
	res = clSetKernelArg(updateQ_kernel, 6, sizeof(float), &theta);
	res = clSetKernelArg(updateQ_kernel, 7, sizeof(int), &width);
	res = clSetKernelArg(updateQ_kernel, 8, sizeof(int), &height);
	
	res = clEnqueueNDRangeKernel(m_queue, updateQ_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
	
	size_t st = width * height * sizeof(float);
	float* gqx = (float*)malloc(st);
	cl_event readEvt;
	status = clEnqueueReadBuffer(m_queue,
		gqxmem,
		CL_FALSE,
		0,
		st,
		gqx,
		0,
		NULL,
		&readEvt);
	status = clFlush(m_queue);
	status = waitForEventAndRelease(&readEvt);
	cv::Mat qx;
	qx.create(1, width*height, CV_32FC1);
	memcpy(qx.data, gqx, st);
	double min = 0, max = 0;
	cv::minMaxIdx(qx, &min, &max);

	res = clSetKernelArg(updateD_kernel, 0, sizeof(cl_mem), &gqxmem);
	res = clSetKernelArg(updateD_kernel, 1, sizeof(cl_mem), &gqymem);
	res = clSetKernelArg(updateD_kernel, 2, sizeof(cl_mem), &dmem);
	res = clSetKernelArg(updateD_kernel, 3, sizeof(cl_mem), &amem);
	res = clSetKernelArg(updateD_kernel, 4, sizeof(float), &theta);
	res = clSetKernelArg(updateD_kernel, 5, sizeof(float), &sigma_d);
	res = clSetKernelArg(updateD_kernel, 6, sizeof(int), &width);
	res = clEnqueueNDRangeKernel(m_queue, updateD_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
}

void RunCL::updateA(int layers, float lambda,float theta)
{
	cl_int status;
	cl_int res;
	cl_event ev;

	res = clSetKernelArg(updateA_kernel, 0, sizeof(cl_mem), &cdatabuf);
	res = clSetKernelArg(updateA_kernel, 1, sizeof(cl_mem), &amem);
	res = clSetKernelArg(updateA_kernel, 2, sizeof(cl_mem), &dmem);
	res = clSetKernelArg(updateA_kernel, 3, sizeof(int), &layers);
	res = clSetKernelArg(updateA_kernel, 4, sizeof(int), &width);
	res = clSetKernelArg(updateA_kernel, 5, sizeof(int), &height);
	res = clSetKernelArg(updateA_kernel, 6, sizeof(float), &lambda);
	res = clSetKernelArg(updateA_kernel, 7, sizeof(float), &theta);

	res = clEnqueueNDRangeKernel(m_queue, updateA_kernel, 1, 0, &global_work_size, 0, 0, NULL, &ev);
}