
#include <limits>
#include <assert.h>

class TInterpolater {
	double *f_x;
	double x_min, x_max, dx, inv_dx;
	unsigned int N;
	
public:
	TInterpolater(const unsigned int _N, const double _x_min, const double _x_max)
		: x_min(_x_min), x_max(_x_max), N(_N), f_x(NULL)
	{
		f_x = new double[N];
		dx = (x_max - x_min) / (double)N;
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
