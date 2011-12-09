#ifndef __INTERPOLATION_H_
#define __INTERPOLATION_H_

#include <math.h>
#include <limits>
#include <assert.h>
#include <cstddef>


// 1-D (linear) interpolation //////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TLinearInterp {
	double *f_x;
	double x_min, x_max, dx, inv_dx;
	unsigned int N;
	
public:
	typedef double (*func1d_t)(double x);
	
	TLinearInterp(double _x_min, double _x_max, unsigned int _N);
	TLinearInterp(func1d_t func, double _x_min, double _x_max, unsigned int _N);
	~TLinearInterp();
	
	double operator()(double x) const;
	double& operator[](unsigned int index);
	
	double get_x(unsigned int index) const;
	double dfdx(double x) const;
	
	void fill(func1d_t func);
};




// 2-D (bilinear) interpolation ////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class TBilinearInterp {
	T *f;				// y is more significant than x, i.e. idx = x + Nx*y
	double x_min, x_max, dx, inv_dx;
	double y_min, y_max, dy, inv_dy;
	double dxdy, inv_dxdy;
	unsigned int Nx, Ny;
	
public:
	typedef T (*func2d_t)(double x, double y);
	typedef T* (*func2d_ptr_t)(double x, double y);
	
	TBilinearInterp(double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	TBilinearInterp(func2d_t func, double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	TBilinearInterp(func2d_ptr_t func, double _x_min, double _x_max, unsigned int Nx, double _y_min, double _y_max, unsigned int Ny);
	~TBilinearInterp();
	
	T operator()(double x, double y) const;
	T& operator[](unsigned int index);
	unsigned int get_index(double x, double y) const;
	void get_xy(unsigned int i, unsigned int j, double &x, double &y) const;
	
	void fill(func2d_t func);
	void fill(func2d_ptr_t func);
};


template<class T>
TBilinearInterp<T>::TBilinearInterp(double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new T[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
}

template<class T>
TBilinearInterp<T>::TBilinearInterp(func2d_t func, double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new T[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
	fill(func);
}

template<class T>
TBilinearInterp<T>::TBilinearInterp(func2d_ptr_t func, double _x_min, double _x_max, unsigned int _Nx, double _y_min, double _y_max, unsigned int _Ny)
	: x_min(_x_min), x_max(_x_max), Nx(_Nx), y_min(_y_min), y_max(_y_max), Ny(_Ny), f(NULL)
{
	f = new T[Nx*Ny];
	dx = (x_max - x_min) / (double)(Nx - 1);
	dy = (y_max - y_min) / (double)(Ny - 1);
	dxdy = dx*dy;
	inv_dx = 1./dx;
	inv_dy = 1./dy;
	inv_dxdy = 1./dxdy;
	fill(func);
}

template<class T>
TBilinearInterp<T>::~TBilinearInterp() {
	delete[] f;
}

template<class T>
unsigned int TBilinearInterp<T>::get_index(double x, double y) const {
	assert((x >= x_min) && (x <= x_max) && (y >= y_min) && (y <= y_max));
	return (unsigned int)((x-x_min)*inv_dx) + (unsigned int)(Nx*(y-y_min)*inv_dy);
}

template<class T>
T TBilinearInterp<T>::operator()(double x, double y) const {
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
	T tmp = inv_dxdy*(f[N00]*(dx-Delta_x)*(dy-Delta_y) + f[N10]*Delta_x*(dy-Delta_y) + f[N01]*(dx-Delta_x)*Delta_y + f[N11]*Delta_x*Delta_y);
	return tmp;
}

template<class T>
T& TBilinearInterp<T>::operator[](unsigned int index) {
	assert(index < Nx*Ny);
	return f[index];
}

template<class T>
void TBilinearInterp<T>::get_xy(unsigned int i, unsigned int j, double &x, double &y) const {
	assert((i < Nx) && (j < Ny));
	x = dx*(double)i - x_min;
	y = dy*(double)j - y_min;
}

template<class T>
void TBilinearInterp<T>::fill(func2d_t func) {
	double x, y;
	for(unsigned int i=0; i<Nx; i++) {
		for(unsigned int j=0; j<Ny; j++) {
			get_xy(i, j, x, y);
			f[i + Nx*j] = func(x, y);
		}
	}
}

template<class T>
void TBilinearInterp<T>::fill(func2d_ptr_t func) {
	double x, y;
	for(unsigned int i=0; i<Nx; i++) {
		for(unsigned int j=0; j<Ny; j++) {
			get_xy(i, j, x, y);
			f[i + Nx*j] = *func(x, y);
		}
	}
}


#endif	// __INTERPOLATION_H_