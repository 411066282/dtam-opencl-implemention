
__kernel void BuildCostVolume(__global float* p,
	__global char3* base,
	__global char3* img,
	__global float* cdata,
	__global float* hdata,
	int layerStep,
	float weight,
	int cols,
	__global float* lo, 
	__global float* hi,
	__global float* a,
	__global float* d,
	int layers)
{
	int xf = get_global_id(0);
	int yf = xf / cols;
	xf = xf % cols;

	unsigned int offset = xf + yf * cols;
	char3 B = base[offset];

	float wi = p[8] * xf + p[9] * yf + p[11];
	float xi = p[0] * xf + p[1] * yf + p[3];
	float yi = p[4] * xf + p[5] * yf + p[7];
	
	float minv = 1000.0, maxv = 0.0, mini = 0;

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	for (unsigned int z = 0; z < layers; z++)
	{
		float c0 = cdata[offset + z*layerStep];
		float w = hdata[offset + z*layerStep];

		float wiz = wi + p[10] * z;
		float xiz = xi + p[2] * z;
		float yiz = yi + p[6] * z;

		int nx = (int)(xiz / wiz);
		int ny = (int)(yiz / wiz);
		unsigned int coff = ny * cols + nx;
		char3 c = img[coff];

		float v1, v2, v3, del, ns;
		float thresh = weight;

		v1 = abs(c.x - B.x);
		v2 = abs(c.y - B.y);
		v3 = abs(c.z - B.z);
		del = v1 + v2 + v3;
		del = fmin(del, thresh)*3.0f / thresh;
		if (c.x + c.y + c.z != 0)
		{
			ns = (c0*w + del) / (w + 1);
			cdata[offset + z*layerStep] = ns;
			hdata[offset + z*layerStep] = w + 1;
		}
		else
		{
			ns = c0;
		}

		if (ns < minv) {
			minv = ns;
			mini = z;
		}
		maxv = fmax(ns, maxv);
	}

	lo[offset] = minv;
	a[offset] = mini;
	d[offset] = mini;
	hi[offset] = maxv;
}
  

 __kernel void CacheG1(__global char* base, __global float* g1p, int cols, int rows)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 int upoff = -(y != 0)*cols;
	 int dnoff = (y < rows-1) * cols;
     int lfoff = -(x != 0);
	 int rtoff = (x < cols-1);

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 unsigned int offset = x + y * cols;

	 float pu, pd, pl, pr;
	 float g0x, g0y, g0, g1;

	 pr = base[offset + rtoff];
	 pl = base[offset + lfoff];
	 pu = base[offset + upoff];
	 pd = base[offset + dnoff];

	 g0x = fabs(pr - pl);
	 g0y = fabs(pd - pu);
	  
	 g0 = fmax(g0x, g0y);
	 g1 = sqrt(g0);
	 g1 = exp(-3.5 * g1);

	 g1p[offset] = g1;
 }

 __kernel void CacheG2(__global float* g1p, __global float* gxp, __global float* gyp, int cols,int rows)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 unsigned int offset = x + y * cols; 
	 
	 int dnoff = (y < rows-1) * cols;
	 int rtoff = (x < cols-1);
	 float g1h, g1l, g1r, g1u, g1d,gx,gy;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 g1h = g1p[offset];
	 g1l = g1h;
	 g1u = g1h;
	 g1d = g1p[offset + dnoff];
	 g1r = g1p[offset + rtoff];
	 gx = fmax(g1l, g1r);
	 gy = fmax(g1u, g1d);
	
	 gxp[offset] = gx;
	 gyp[offset] = gy;
 }

 __kernel void UpdateQ(__global float* gxpt,
	 __global float* gypt, 
	 __global float* gqxpt,
	 __global float* gqypt,
	 __global float* dpt,
	 float epsilon,
	 float sigma_q,
	 int cols,
	 int rows)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 int dnoff = (y < rows-1) * cols;
	 int rtoff = (x < cols-1);
	 unsigned int pt = x + y * cols;

	 barrier(CLK_GLOBAL_MEM_FENCE);
	
	 float dh = dpt[pt];
	 float gqx, gqy, dr, dd, qx, qy, gx, gy;

	 gqx = gqxpt[pt];
	 gx = gxpt[pt] + .005f;
	 dr = dpt[pt + rtoff];
	 qx = gqx / gx;
	 qx = (qx + sigma_q*gx*(dr - dh)) / (1 + sigma_q*epsilon);
	 qx = qx / fmax(1.0f, fabs(qx));
	 gqx = gx *  qx;
	 gqxpt[pt] = gqx;

	 gqy = gqypt[pt];
	 gy = gypt[pt] + .005f;
	 dd = dpt[pt + dnoff];
	 qy = gqy / gy;
	 qy = (qy + sigma_q*gy*(dd - dh)) / (1 + sigma_q*epsilon);
	 qy = qy / fmax(1.0f, fabs(qy));
	 gqy = gy *  qy;
	 gqypt[pt] = gqy;
 }

 __kernel void UpdateD(__global float* gqxpt,
	 __global float* gqypt,
	 __global float* dpt,
	 __global float* apt,
	 float theta,
	 float sigma_d,
	 int cols)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 int upoff = -(y != 0)*cols;
	 int lfoff = -(x != 0);

	 unsigned int pt = x + y * cols;

	 float dacc;
	 float gqr, gql;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 gqr = gqxpt[pt];
	 gql = gqxpt[pt + lfoff];
	 dacc = gqr - gql;
	
	 float gqu, gqd;
	 float d = dpt[pt];
	 float a = apt[pt];
	 gqd = gqypt[pt];
	 gqu = gqypt[pt + upoff];
	 if (y == 0)gqu = 0;
	 dacc += gqd - gqu;
	
	 d = (d + sigma_d*(dacc + a / theta)) / (1 + sigma_d / theta);

	 dpt[pt] = d;
 }

 float afunc(float data, float theta, float d, float ds, int a, float lambda)
 {
	 return 1.0 / (2.0*theta)*ds*ds*(d - a)*(d - a) + data * lambda;
 }

 __kernel void UpdateA(__global float* cdata,
	__global float* a,
	__global float* d,
	int layers,
	int cols,
	int rows,
	float lambda,
	float theta)
 {
	 int x = get_global_id(0);
	 int y = x / cols;
	 x = x % cols;
	 unsigned int pt = x + y * cols;

	 barrier(CLK_GLOBAL_MEM_FENCE);

	 float dv = d[pt];

	 float depthStep = 1.0f / layers;
	 float vlast, vnext, v, A, B, C;

	 unsigned int mini = 0;
	 unsigned int layerstep = rows * cols;

	 float minv = v = afunc(cdata[pt], theta, dv, depthStep, 0, lambda);

	 vnext = afunc(cdata[pt + layerstep], theta, dv, depthStep, 1, lambda);

	 for (unsigned int z = 2; z<layers; z++) {
		 vlast = v;
		 v = vnext;
		 vnext = afunc(cdata[pt + z * layerstep], theta, dv, depthStep, z, lambda);
		 if (v<minv) {
			 A = vlast;
			 C = vnext;
			 minv = v;
			 mini = z - 1;
		 }
	 }

	 if (vnext<minv) {//last was best
		 a[pt] = layers - 1;
		 return;
	 }

	 if (mini == 0) {//first was best
		 a[pt] = 0;
		 return;
	 }

	 B = minv;//avoid divide by zero, since B is already <= others, make < others

	 float denom = (A - 2 * B + C);
	 float delt = (A - C) / (denom * 2);

	 if (denom != 0)
		 a[pt] = delt + (float)mini;
	 else
		 a[pt] = mini;
 }