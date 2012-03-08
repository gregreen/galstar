
#include "chain.h"

/*************************************************************************
 *   Chain Class Member Functions
 *************************************************************************/

TChain::TChain(unsigned int _N, unsigned int _capacity)
	: stats(_N)
{
	N = _N;
	length = 0;
	total_weight = 0;
	set_capacity(_capacity);
}

TChain::~TChain() {}

void TChain::add_point(double* element, double L_i, double w_i) {
	stats(element, (unsigned int)w_i);
	for(unsigned int i=0; i<N; i++) {
		x.push_back(element[i]);
	}
	L.push_back(L_i);
	w.push_back(w_i);
	total_weight += w_i;
	length += 1;
}

void TChain::clear() {
	x.clear();
	L.clear();
	w.clear();
	stats.clear();
	total_weight = 0;
	length = 0;
}

void TChain::set_capacity(unsigned int _capacity) {
	capacity = _capacity;
	x.reserve(N*capacity);
	L.reserve(capacity);
	w.reserve(capacity);
}

unsigned int TChain::get_capacity() const {
	return capacity;
}

unsigned int TChain::get_length() const {
	return length;
}

double TChain::get_total_weight() const {
	return total_weight;
}

double* TChain::get_element(unsigned int i) {
	return &(x[i*N]);
}

double TChain::get_L(unsigned int i) const {
	return L[i];
}

double TChain::get_w(unsigned int i) const {
	return w[i];
}

void TChain::append(TChain& chain, bool reweight, double nsigma) {
	assert(chain.N == N);	// Make sure the two chains have the same dimensionality
	
	// Append the last chain to this one
	if(capacity < length + chain.length) { set_capacity(1.5*(length + chain.length)); }
	std::vector<double>::iterator w_end_old = w.end();
	x.insert(x.end(), chain.x.begin(), chain.x.end());
	L.insert(L.end(), chain.L.begin(), chain.L.end());
	w.insert(w.end(), chain.w.begin(), chain.w.end());
	
	// Weight each chain according to Bayesian evidence, if requested
	double a = 1.;
	if(reweight) {
		a = chain.get_Z_harmonic(nsigma) / get_Z_harmonic(nsigma) * total_weight / chain.total_weight;
		std::vector<double>::iterator w_end = w.end();
		for(std::vector<double>::iterator it = w_end_old; it != w_end; ++it) {
			*it *= a;
		}
	}
	
	stats += a * chain.stats;
	length += chain.length;
	total_weight += a * chain.total_weight;
}

double* TChain::operator [](unsigned int i) {
	return &(x[i*N]);
}

void TChain::operator +=(TChain& rhs) {
	append(rhs);
}

double TChain::get_Z_harmonic(double nsigma) {
	// Get the covariance and determinant of the chain
	gsl_matrix* Sigma = gsl_matrix_alloc(N, N);
	gsl_matrix* invSigma = gsl_matrix_alloc(N, N);
	double detSigma;
	stats.get_cov_matrix(Sigma, invSigma, &detSigma);
	
	// Get the mean from the stats class
	double* mu = new double[N];
	for(unsigned int i=0; i<N; i++) {
		mu[i] = stats.mean(i);
	}
	
	// Determine the volume normalization (the prior volume)
	double V = 1. / sqrt(detSigma) * 2. * pow(SQRTPI * nsigma, (double)N) / (double)(N) / gsl_sf_gamma((double)(N)/2.);
	
	// Determine <1/L> inside the prior volume
	double dist2;
	double sum_invL = 0;
	//double sum_weights = 0;
	for(unsigned int i=0; i<length; i++) {
		dist2 = metric_dist2(invSigma, get_element(i), mu, N);
		//std::cout << dist2 << std::endl;
		if(dist2 < nsigma*nsigma) {
			sum_invL += w[i] / exp(L[i]);
			//sum_weights += w[i];
		}
	}
	
	// Cleanup
	gsl_matrix_free(Sigma);
	gsl_matrix_free(invSigma);
	delete[] mu;
	
	// Return the estimate of Z
	return V / sum_invL * total_weight;
}


/*double vect_dist(double* x_1, double *x_2, unsigned int N) {
	double tmp = 0;
	for(unsigned int i=0; i<N; i++) {
		tmp += (x_1[i] - x_2[i]) * (x_1[i] - x_2[i]);
	}
	return sqrt(tmp);
}*/