#ifndef __INTERPOLATER_H_
#define __INTERPOLATER_H_

#include <math.h>
#include <limits>
#include <assert.h>


// 1-D interpolation //////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TInterpolater {
	double *f_x;
	double x_min, x_max, dx, inv_dx;
	unsigned int N;
	
public:
	TInterpolater(const unsigned int _N, const double _x_min, const double _x_max)
		: x_min(_x_min), x_max(_x_max), N(_N), f_x(NULL)
	{
		f_x = new double[N];
		dx = (x_max - x_min) / (double)(N - 1);
		inv_dx = 1./dx;
	}
	
	~TInterpolater() { delete[] f_x; }
	
	double operator()(const double x) const;
	double& operator[](const unsigned int index);
	
	double get_x(const unsigned int index) const;
	double dfdx(const double x) const;
};

inline double TInterpolater::operator()(const double x) const {
	if((x < x_min) || (x > x_max)) { return std::numeric_limits<double>::quiet_NaN(); }
	double index_dbl = (x - x_min) * inv_dx;
	unsigned int index_nearest = (unsigned int)(index_dbl + 0.5);
	double dist = (index_dbl - (double)(index_nearest)) * dx;
	if(index_nearest == N) { return f_x[N-1] - dist*(f_x[N-1]-f_x[N-2])*inv_dx; }
	if(dist == 0) {
		return f_x[index_nearest];
	} else if(dist > 0) {
		return f_x[index_nearest] + dist * (f_x[index_nearest+1]-f_x[index_nearest])*inv_dx;
	} else {
		return f_x[index_nearest] + dist * (f_x[index_nearest]-f_x[index_nearest-1])*inv_dx;
	}
}

inline double TInterpolater::dfdx(const double x) const {
	if((x < x_min) || (x > x_max)) { return std::numeric_limits<double>::quiet_NaN(); }
	double index_dbl = (x - x_min) * inv_dx;
	unsigned int index_nearest = (unsigned int)(index_dbl + 0.5);
	if(index_nearest == 0) {
		return (f_x[1]-f_x[0])*inv_dx;
	} else if((index_nearest == N-1) || (index_nearest == N)) {
		return (f_x[N-1]-f_x[N-2])*inv_dx;
	}
	double diff = index_dbl - (double)(index_nearest);
	if(diff >= 0) {
		return (f_x[index_nearest+1]-f_x[index_nearest])*inv_dx;
	} else {
		return (f_x[index_nearest]-f_x[index_nearest-1])*inv_dx;
	}
}

inline double TInterpolater::get_x(const unsigned int index) const {
	assert(index < N);
	return x_min + (double)index * dx;
}

inline double& TInterpolater::operator[](const unsigned int index) {
	assert(index < N);
	return f_x[index];
}




// 2-D interpolation //////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TBilinearInterp {
	double *f;				// y is more significant than x, i.e. idx = x + Nx*y
	double x_min, x_max, dx, inv_dx;
	double y_min, y_max, dy, inv_dy;
	double dxdy, inv_dxdy;
	unsigned int Nx, Ny;
	
public:
	typedef double (*func2d_t)(double x, double y);
	
	TBilinearInterp(double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	TBilinearInterp(func2d_t func, double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	~TBilinearInterp();
	
	double operator()(double x, double y) const;
	double& operator[](unsigned int index);
	unsigned int get_index(double x, double y) const;
	void get_xy(unsigned int i, unsigned int j, double &x, double &y) const;
	
	void fill(func2d_t func);
};


TBilinearInterp::TBilinearInterp(double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new double[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
}

TBilinearInterp::TBilinearInterp(func2d_t func, double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new double[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
	fill(func);
}

TBilinearInterp::~TBilinearInterp() {
	delete[] f;
}

inline unsigned int TBilinearInterp::get_index(double x, double y) const {
	assert((x >= x_min) && (x <= x_max) && (y >= y_min) && (y <= y_max));
	return (unsigned int)((x-x_min)*inv_dx) + (unsigned int)(Nx*(y-y_min)*inv_dy);
}

inline double TBilinearInterp::operator()(double x, double y) const {
	double idx = floor((x-x_min)*inv_dx);
	assert((idx >= 0) && (idx < Nx));
	double idy = floor((y-y_min)*inv_dy);
	assert((idy >= 0) && (idy < Ny));
	double Delta_x = x - x_min - dx*idx;
	double Delta_y = y - y_min - dy*idy;
	unsigned int N00 = (unsigned int)idx + Nx*(unsigned int)idy;
	unsigned int N10 = N00 + 1;
	unsigned int N01 = N00 + Nx;
	unsigned int N11 = N00 + 1 + Nx;
	return inv_dxdy*(f[N00]*(dx-Delta_x)*(dy-Delta_y) + f[N10]*Delta_x*(dy-Delta_y) + f[N01]*(dx-Delta_x)*Delta_y + f[N11]*Delta_x*Delta_y);
}

inline double& TBilinearInterp::operator[](unsigned int index) {
	assert(index < Nx*Ny);
	return f[index];
}

inline void TBilinearInterp::get_xy(unsigned int i, unsigned int j, double &x, double &y) const {
	assert((i < Nx) && (j < Ny));
	x = dx*(double)i - x_min;
	y = dy*(double)j - y_min;
}

inline void TBilinearInterp::fill(func2d_t func) {
	double x, y;
	for(unsigned int i=0; i<Nx; i++) {
		for(unsigned int j=0; j<Ny; j++) {
			get_xy(i, j, x, y);
			f[i + Nx*j] = func(x, y);
		}
	}
}

#endif	// __INTERPOLATER_H_