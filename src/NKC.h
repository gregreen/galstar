
#ifndef _NKC_H__
#define _NKC_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <limits>
#include <assert.h>
#include <omp.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

#include <boost/cstdint.hpp>

#include "stats.h"

//uint64_t fail_count;
//uint64_t hit_count;

/*************************************************************************
 *   Prototypes
 *************************************************************************/

double fast_exp(double y);

void seed_gsl_rng(gsl_rng **r);

void Gaussian_proposal(const double *const x_0, double *x_1, const double *const transform, unsigned int N, const double bandwidth, gsl_rng *r);

double Gaussian_density(const double *const x, const double *mu, const double *const inv_cov, unsigned int N, double norm, double bandwidth, double bandwidth_N, double log_bandwidth, bool return_log);

static void calc_transform(gsl_matrix *transform, const gsl_matrix * const covariance);

struct TStats;

static void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R);

/*************************************************************************
 *   Normal Kernel Coupler class protoype
 *************************************************************************/

// The Normal Kernel Coupler behaves like a Metropolis-Hastings Markov Chain, with the
// difference that each link in the chain is an ensemble of states. At each step, one
// component state in the ensemble is chosen to step, and a proposal is drawn from a
// multivariate normal distribution centered on another randomly chosen component state
// in the ensemble. The standard MH algorithm is then applied to determine whether the
// proposal is accepted or rejected. If the proposal is accepted, the chosen state transitions,
// while all other states in the ensemble remain in their current positions.
// The stength of the Normal Kernel Coupler is that it allows transitions between local
// minima in a multimodal target distribution. As long as some of the component states lie
// in each local minimum, the component states will continue to transition between even
// widely-separated minima.
template<class TParams, class TLogger>
class TNKC {
	struct TState;
	TState* X;		// Current state
	unsigned int N;		// Dimensionality of parameter space
	unsigned int size;	// Number of component states in overall state
	
	TParams& params;	// Constant model parameters
	
	TStats stats;		// Stores expectation values, covariance, etc.
	TLogger& logger;	// Object which logs states in the chain
	TState X_ML;		// Maximum likelihood point encountered
	
	TState Y;		// Proposal state
	double *V;		// Proposal covariance
	double *sqrtV;		// Spectral decomposition of V. sqrtV sqrtV^T = V
	double *invV;		// V^{-1}
	double normV, twopiN;	// Normalization of proposal pdf
	double h, h_N, log_h;	// Bandwidth tuning constant
	double tune_rate;	// (tune_rate > 1) Rate at which bandwidth grows or shrinks during burn-in
	unsigned int i_updt;	// Index of next state to update
	gsl_rng* r;		// Random number generator
	
	bool use_log;		// If true, <pdf> returns log(pi(X)). Default value is <true>.
	
	double q_YX();			// q(Y | X)
	double q_XY(unsigned int u);	// q(X_u | Y , X_(-u))
	void get_proposal();		// Generate a proposal state
	void get_proposal(unsigned int u);	// Generate a proposal state centered on X_u
	void update_sqrtV();		// Calculate sqrt(V) from V, the proposal covariance
	boost::uint64_t N_accepted, N_rejected;	// # of steps which have been accepted and rejected. Used to tune and track acceptance rate.
	
	// Cache which stores the results of calls to Gaussian_density
	//struct TGaussianDensityCache;
	//TGaussianDensityCache GDCache;
	
public:
	typedef double (*pdf_t)(const double *const _X, unsigned int _N, TParams& _params);
	typedef void (*rand_state_t)(double *const _X, unsigned int _N, gsl_rng* r, TParams& _params);
	
	// Constructor & destructor
	TNKC(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _size, const double *const sigma, TParams& _params, TLogger& _logger);
	~TNKC();
	
	// Mutators
	void step(bool log_step, bool jump);		// Take one step
	void burn_in(unsigned int N_rounds, unsigned int round_length, double target_acceptance, bool tune, bool adjust_proposal);	// Burn in while adjusting proposal covariance
	void set_scale(const double *const sigma);								// Set proposal covariance to diag(sigma)
	void set_covariance(const double *const cov);								// Set proposal covariance to <cov>
	void set_bandwidth(double _h) { h = _h; h_N = pow(h,N); log_h = log(_h); };	// Set bandwidth by which the step size is scaled
	void set_tune_rate(double _tune_rate) { tune_rate = _tune_rate; }		// Set rate at which bandwidth is tuned during burn-in
	void set_use_log(bool _use_log) { use_log = _use_log; }				// Set whether <pdf> returns log(pi(X))
	void flush(bool log_steps);							// Log the outstanding component states
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return stats; }
	double get_bandwidth() { return h; }
	void print_state();
	double get_acceptance_rate() { return (double)N_accepted/(double)(N_accepted+N_rejected); }
	
private:
	rand_state_t rand_state;	// Function which generates a random state
	pdf_t pdf;			// pi(X), a function proportional to the target distribution
};


/*************************************************************************
 *   Constructor and destructor
 *************************************************************************/

// Constructor
// Inputs:
// 	_pdf		Target distribution, up to a normalization constant
// 	_rand_state	Function which generates a random state, used for initialization of the chain
// 	_size		# of concurrent states in the ensemble
// 	sigma[N]	Initial std. dev. of proposal distribution in each direction
// 	_params		Misc. constant model parameters needed by _pdf
// 	_logger		Object which logs the chain in some way. It must have an operator()(double state[N], unsigned int weight).
// 			The logger could, for example, bin the chain, or just push back each state into a vector.
template<class TParams, class TLogger>
TNKC<TParams, TLogger>::TNKC(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _size, const double *const sigma, TParams& _params, TLogger& _logger)
	: pdf(_pdf), rand_state(_rand_state), params(_params), logger(_logger), N(_N), size(_size), X(NULL), Y(_N), r(NULL), V(NULL), sqrtV(NULL), invV(NULL), stats(_N)
{
	// Seed the random number generator
	seed_gsl_rng(&r);
	
	// Generate the initial state and record the most likely point
	X = new TState[size];
	for(unsigned int i=0; i<size; i++) {
		X[i].initialize(N);
	}
	unsigned int index_of_best = 0;
	for(unsigned int i=0; i<size; i++) {
		rand_state(X[i].element, N, r, params);
		X[i].pi = pdf(X[i].element, N, params);
		X[i].weight = 1;
		if(X[i] > X[index_of_best]) { index_of_best = i; }
	}
	X_ML = X[index_of_best];
	
	// Allocate V, sqrt(V)
	V = new double[N*N];
	sqrtV = new double[N*N];
	invV = new double[N*N];
	
	// Set the initial proposal covariance
	set_scale(sigma);
	
	// Set the bandwidth tuning parameter to its default value
	set_bandwidth(sqrt(1.4 / pow((double)size, 2./(4.+(double)N))));
	//log_h = log(h);
	tune_rate = 1.015;
	twopiN = pow(6.28318531, (double)N);
	
	i_updt = 0;
	
	// Initialize the statistics object
	//stats.clear();
	
	// Assume that the function <pdf> returns the log probability
	use_log = true;
	
	N_accepted = 0;
	N_rejected = 0;
}

template<class TParams, class TLogger>
TNKC<TParams, TLogger>::~TNKC() {
	gsl_rng_free(r);
	if(X != NULL) { delete[] X; X = NULL; }
	if(V != NULL) { delete[] V; V = NULL; }
	if(sqrtV != NULL) { delete[] sqrtV; sqrtV = NULL; }
	if(invV != NULL) { delete[] invV; invV = NULL; }
}


/*************************************************************************
 *   Structs
 *************************************************************************/

// Component state type
template<class TParams, class TLogger>
struct TNKC<TParams, TLogger>::TState {
	double *element;
	unsigned int N;
	double pi;		// pdf(X) = likelihood of state (up to normalization)
	unsigned int weight;	// # of times the chain has remained on this state
	
	TState() : N(0), element(NULL) {}
	TState(unsigned int _N) : N(_N) { element = new double[N]; }
	~TState() { if(element != NULL) { delete[] element; } }
	
	void initialize(unsigned int _N) {
		N = _N;
		if(element == NULL) { element = new double[N]; }
	}
	
	double& operator[](unsigned int index) { return element[index]; }
	
	// Assignment operator
	TState& operator=(const TState& rhs) {
		for(unsigned int i=0; i<N; i++) { element[i] = rhs.element[i]; }
		pi = rhs.pi;
		weight = rhs.weight;
		return *this;
	}
	
	// Compares everything but weight
	bool operator==(const TState& rhs) {
		assert(rhs.N == N);
		if(pi != rhs.pi){ return false; }
		for(unsigned int i=0; i<N; i++) { if(element[i] != rhs.element[i]) { return false; } }
		return true;
	}
	bool operator!=(const TState& rhs) {
		assert(rhs.N == N);
		if(pi != rhs.pi){ return true; }
		for(unsigned int i=0; i<N; i++) { if(element[i] != rhs.element[i]) { return true; } }
		return false;
	}
	
	// The operators > and < compare the likelihood of two states
	bool operator>(const TState& rhs) { return pi > rhs.pi; }
	bool operator<(const TState& rhs) { return pi < rhs.pi; }
	bool operator>(const double& rhs) { return pi > rhs; }
	bool operator<(const double& rhs) { return pi < rhs; }
};

/*************************************************************************
 *   Private functions
 *************************************************************************/

// q(Y | X)
template<class TParams, class TLogger>
inline double TNKC<TParams, TLogger>::q_YX() {
	double tmp = 0.;
	for(unsigned int i=0; i<size; i++) { tmp += Gaussian_density(Y.element, X[i].element, invV, N, normV, h, h_N, log_h, false); }//fail_count++; }
	return tmp / (double)size;
}

// q(X_u | Y , X_(-u))
template<class TParams, class TLogger>
inline double TNKC<TParams, TLogger>::q_XY(unsigned int u) {
	double tmp = 0.;
	for(unsigned int i=0; i<size; i++) {
		if(i != u) { tmp += Gaussian_density(X[u].element, X[i].element, invV, N, normV, h, h_N, log_h, false); }
		//if(i != u) { tmp += GDCache(u, i, X[u], X[i], this); }
	}
	tmp += Gaussian_density(X[u].element, Y.element, invV, N, normV, h, h_N, log_h, false);
	//fail_count++;
	return tmp / (double)size;
}

// Generate a proposal state
template<class TParams, class TLogger>
inline void TNKC<TParams, TLogger>::get_proposal() {
	unsigned int u = gsl_rng_uniform_int(r, (long unsigned int)size);
	Gaussian_proposal(X[u].element, Y.element, sqrtV, N, h, r);
	Y.pi = pdf(Y.element, N, params);
	Y.weight = 0;
}

// Generate a proposal state from a specified component state X_u
template<class TParams, class TLogger>
inline void TNKC<TParams, TLogger>::get_proposal(unsigned int u) {
	Gaussian_proposal(X[u].element, Y.element, sqrtV, N, h, r);
	Y.pi = pdf(Y.element, N, params);
	Y.weight = 0;
}

// Calculate sqrt(V), V^{-1} and normV from V, the proposal covariance
template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::update_sqrtV() {
	// Copy V, the sample covariance, into a gsl_matrix
	gsl_matrix* V_mat = gsl_matrix_alloc(N, N);
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) { gsl_matrix_set(V_mat, i, j, V[N*i+j]); }
	}
	
	// Calculate sqrt(V) (this means that sqrtV sqrtV^T = V)
	gsl_matrix* tmp = gsl_matrix_alloc(N, N);
	calc_transform(tmp, V_mat);
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) { sqrtV[N*i+j] = gsl_matrix_get(tmp, i, j); }
	}
	
	// Calculate V^{-1}
	int s;
	gsl_permutation* p = gsl_permutation_alloc(N);
	gsl_matrix* LU = gsl_matrix_alloc(N, N);
	gsl_matrix_memcpy(LU, V_mat);
	gsl_linalg_LU_decomp(LU, p, &s);
	gsl_linalg_LU_invert(LU, p, tmp);
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) { invV[N*i+j] = gsl_matrix_get(tmp, i, j); }
	}
	
	// Calculate normV = 1 / sqrt( det(V) (2pi)^N )
	normV = 1./ sqrt(fabs(gsl_linalg_LU_det(LU, s)) * twopiN);
	//if(use_log) { normV = log(normV); }
	
	// Cleanup
	gsl_matrix_free(LU);
	gsl_matrix_free(tmp);
	gsl_matrix_free(V_mat);
	gsl_permutation_free(p);
}

/*************************************************************************
 *   Mutators
 *************************************************************************/

template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::step(bool log_step=true, bool jump=true) {
	if(jump) { get_proposal(); } else { get_proposal(i_updt); }
	
	if(Y.pi > X_ML.pi) { X_ML = Y; }
	
	double alpha;
	if(isinff(X[i_updt].pi) && !(isinff(Y.pi))) {
		alpha = 2.;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
	} else {
		if(use_log) { alpha = fast_exp(Y.pi - X[i_updt].pi); } else { alpha = Y.pi / X[i_updt].pi; }
		if(jump && (alpha != 0)) {
			double q_factor = q_XY(i_updt) / q_YX();
			if(gsl_rng_uniform(r) < 0.00001) { std::cout << q_factor << "\t" << alpha<< std::endl; }
			//if(q_factor != 1) { std::cout << q_factor << std::endl; }
			alpha *= q_factor;
		}
	}
	
	bool accept = false;
	if(alpha > 1.) {
		accept = true;
	} else {
		double p = gsl_rng_uniform(r);
		if(p == 0.) {
			if(use_log && (Y > -std::numeric_limits<double>::infinity())) {
				accept = true;
			} else if(!use_log && (Y > 0.)) {
				accept = true;
			}
		} else if(p < alpha) {
			accept = true;
		}
	}
	if(accept) {
		stats(X[i_updt].element, X[i_updt].weight);
		if(log_step) {
			#pragma omp critical (logger)
			logger(X[i_updt].element, X[i_updt].weight);
		}
		X[i_updt] = Y;
		N_accepted++;
	} else {
		N_rejected++;
	}
	
	for(unsigned int i=0; i<size; i++) { X[i].weight++; }
	
	// Increment the index of the state to be updated
	i_updt++;
	if(i_updt >= size) { i_updt = 0; }
}

template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::burn_in(unsigned int N_rounds, unsigned int round_length, double target_acceptance=0.25, bool tune=true, bool adjust_proposal=true) {
	bool jump = false;
	for(unsigned int i=0; i<N_rounds; i++) {
		if(i > N_rounds/2) {
			jump = true;
		}
		for(unsigned int n=0; n<round_length; n++) { step(false, jump); }
		flush(false);
		for(unsigned int j=0; j<N; j++) {
			for(unsigned int k=j+1; k<N; k++) { V[N*j+k] = stats.cov(j,k); V[k+N*j] = V[N*j+k]; }
			V[N*j+j] = stats.cov(j,j);
		}
		if(adjust_proposal) { update_sqrtV(); }
		if(i < N_rounds/3) {
			stats.clear();
		} else if (tune) {
			double acceptance_rate = get_acceptance_rate();
			if(acceptance_rate < target_acceptance/1.2) {
				set_bandwidth(h/tune_rate);
				//#pragma omp critical (cerr)
				//std::cerr << "# Thread " << omp_get_thread_num() << " bandwidth <<< " << get_bandwidth() << " (acceptance = " << acceptance_rate << " , det = " << twopiN/(normV*normV)/pow(h,N) << ")" << std::endl;
			} else if(acceptance_rate > target_acceptance*1.2) {
				set_bandwidth(h*tune_rate);
				//#pragma omp critical (cerr)
				//std::cerr << "# Thread " << omp_get_thread_num() << " bandwidth >>> " << get_bandwidth() << " (acceptance = " << acceptance_rate << " , det = " << twopiN/(normV*normV)/pow(h,N) << ")" << std::endl;
			}
			N_accepted = 0;
			N_rejected = 0;
		}
	}
	stats.clear();
	for(unsigned int i=0; i<size; i++) { X[i].weight = 1; }
	N_accepted = 0;
	N_rejected = 0;
}

// Set proposal covariance to diag(sigma)
template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::set_scale(const double *const sigma) {
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=i+1; j<N; j++) { V[N*i+j] = 0.; V[N*j+i] = 0.; }
		V[N*i+i] = sigma[i]*sigma[i];
	}
	update_sqrtV();
	//GDCache.clear();
}

// Set proposal covariance to <cov>
template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::set_covariance(const double *const cov) {
	for(unsigned int i=0; i<N*N; i++) {
		V[i] = cov[i];
		//for(unsigned int j=i+1; j<N; j++) { V[N*i+j] = cov[N*i+j]; V[N*j+i] = cov[N*i+j]; }
		//V[N*i+i] = cov[N*i+i];
	}
	update_sqrtV();
	//GDCache.clear();
}

template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::flush(bool log_steps=true) {
	for(unsigned int i=0; i<size; i++) {
		if(log_steps) {
			#pragma omp critical (logger)
			logger(X[i].element, X[i].weight);
		}
		stats(X[i].element, X[i].weight);
		X[i].weight = 0;
	}
}


/*************************************************************************
 *   Accessors
 *************************************************************************/

template<class TParams, class TLogger>
void TNKC<TParams, TLogger>::print_state() {
	for(unsigned int i=0; i<size; i++) {
		std::cout << "p(X) = " << X[i].pi << std::endl;
		std::cout << "Weight = " << X[i].weight << std::endl << "X [" << i << "] = { ";
		for(unsigned int j=0; j<N; j++) { std::cout << (j == 0 ? "" : " ") << std::setprecision(3) << X[i].element[j]; }
		std::cout << " }" << std::endl << std::endl;
	}
}


/*************************************************************************
 *   Auxiliary Functions
 *************************************************************************/

/// See paper "A Fast, Compact Approximation of the Exponential Function".
/// 2x to 9x faster than exp(x)!
/// Can be off by about +-4% in the range -100 to 100.
///
/// On Intel Core2 Quad Q9550, VS2008 SP1, evaluations per second:
///  20035805 exp(x) with /fp:fast
///  29961267 exp(x) with /fp:precise
///  30386769 exp(x) with /arch:SSE2 /fp:precise
///  92379955 exp(x) with /arch:SSE2 /fp:fast
/// 132160844 fast_exp(x) with /fp:precise
/// 140163862 fast_exp(x) with /fp:fast
/// 172233728 fast_exp(x) with /arch:SSE2 /fp:precise
/// 182784751 fast_exp(x) with /arch:SSE2 /fp:fast
inline double fast_exp(double y) {
	if(y < -700.) {
		return 0.;
	} else if(y > 700.) {
		return std::numeric_limits<double>::infinity();
	}
	double d;
	*((int*)(&d) + 0) = 0;
	*((int*)(&d) + 1) = (int)(1512775 * y + 1072632447);
	return d;
}

// Seed a gsl_rng with the Unix time in nanoseconds
inline void seed_gsl_rng(gsl_rng **r) {
	timespec t_seed;
	clock_gettime(CLOCK_REALTIME, &t_seed);
	long unsigned int seed = 1e9*(long unsigned int)t_seed.tv_sec;
	seed += t_seed.tv_nsec;
	*r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(*r, seed);
}

// Sample from a Gaussian proposal distribution
inline void Gaussian_proposal(const double *const x_0, double *x_1, const double *const transform, unsigned int N, const double bandwidth, gsl_rng *r) {
	for(unsigned int i=0; i<N; i++) { x_1[i] = x_0[i]; }
	double tmp;
	for(unsigned int j=0; j<N; j++) {
		tmp = gsl_ran_gaussian_ziggurat(r, 1.);
		for(unsigned int i=0; i<N; i++) { x_1[i] += bandwidth * transform[N*i+j] * tmp; }
	}
}

// Get the density Gaussian proposal distribution
inline double Gaussian_density(const double *const x, const double *const mu, const double *const inv_cov, unsigned int N, double norm, double bandwidth=1., double bandwidth_N=1., double log_bandwidth=0., bool return_log=false) {
	double tmp = 0.;
	double w;
	for(unsigned int i=0; i<N; i++) {
		w = 0.;
		for(unsigned int j=0; j<N; j++) { w += (x[j] - mu[j]) * inv_cov[N*i+j]; }
		tmp += w * (x[i] - mu[i]);
	}
	double exponent = -tmp/(2.*bandwidth*bandwidth);
	// p(X) = 1/(h norm) exp( - tmp / (2 h^2) )
	if(return_log) { return -N*log_bandwidth + norm + exponent; } else { return norm / bandwidth_N * fast_exp(exponent); } // /pow(bandwidth,N)
}


// Calculate the transformation matrix A s.t. AA^T = S, where S is the covariance matrix.
static void calc_transform(gsl_matrix *transform, const gsl_matrix * const covariance) {
	assert(covariance->size1 == covariance->size2);
	assert(transform->size1 == transform->size2);
	assert(covariance->size1 == transform->size1);
	size_t N = covariance->size1;
	// Calculate the eigendecomposition of the covariance matrix
	gsl_matrix_memcpy(transform, covariance);
	gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(N);
	gsl_vector* eival = gsl_vector_alloc(N);
	gsl_matrix* eivec = gsl_matrix_alloc(N, N);
	gsl_eigen_symmv(transform, eival, eivec, w);
	gsl_matrix* sqrt_eival = gsl_matrix_calloc(N, N);
	double tmp;
	for(size_t i=0; i<N; i++) {
		tmp = gsl_vector_get(eival, i);
		gsl_matrix_set(sqrt_eival, i, i, sqrt(fabs(tmp)));
		if(tmp < 0.) {
			for(size_t j=0; j<N; j++) { gsl_matrix_set(eivec, j, i, -gsl_matrix_get(eivec, j, i)); }
		}
	}
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., eivec, sqrt_eival, 0., transform);
	// Cleanup
	gsl_matrix_free(sqrt_eival);
	gsl_matrix_free(eivec);
	gsl_vector_free(eival);
	gsl_eigen_symmv_free(w);
}




template<class TParams, class TLogger>
class TParallelNKC {
	TNKC<TParams, TLogger>** nkc;
	unsigned int N;
	unsigned int N_chains;
	TStats stats;
	TStats** component_stats;
	TLogger& logger;
	TParams& params;
	double *R;
	
public:
	// Constructor & Destructor
	TParallelNKC(typename TNKC<TParams, TLogger>::pdf_t _pdf, typename TNKC<TParams, TLogger>::rand_state_t _rand_state, unsigned int _N, unsigned int _size, const double *const sigma, TParams& _params, TLogger& _logger, unsigned int _N_chains);
	~TParallelNKC();
	
	// Mutators - TODO: Parallelize some of these routines, using <burn_in> as a template
	void step(unsigned int N_steps, bool log_step);				// Take the given number of steps in each NKC object
	void burn_in(unsigned int N_rounds, unsigned int round_length, double target_acceptance, bool tune, bool adjust_proposal);	// Burn in while adjusting proposal covariance in each NKC object
	void set_scale(const double *const sigma) { for(unsigned int i=0; i<N_chains; i++) { nkc[i]->set_scale(sigma); } };		// Set proposal covariance to diag(sigma)
	void set_covariance(const double *const cov) { for(unsigned int i=0; i<N_chains; i++) { nkc[i]->set_covariance(cov); } };	// Set proposal covariance to <cov>
	void set_bandwidth(double h) { for(unsigned int i=0; i<N_chains; i++) { nkc[i]->set_bandwidth(h); } }				// Set bandwidth by which the step size is scaled
	void set_tune_rate(double _tune_rate) { for(unsigned int i=0; i<N_chains; i++) { nkc[i]->set_tune_rate(_tune_rate); } }		// Set rate at which bandwidth is tuned during burn-in
	void set_use_log(bool use_log) { for(unsigned int i=0; i<N_chains; i++) { nkc[i]->set_use_log(use_log); } }			// Set whether <pdf> returns log(pi(X))
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return stats; }
	TStats& get_stats(unsigned int index) { assert(index < N_chains); return nkc[index]->get_stats(); }
	void get_GR_diagnostic(double *const GR) { for(unsigned int i=0; i<N; i++) { GR[i] = R[i]; } }
	double get_GR_diagnostic(unsigned int index) { return R[index]; }
	double get_bandwidth(unsigned int index) { assert(index < N_chains); return nkc[index]->get_bandwidth(); }
	void print_stats();
	void print_state() { for(unsigned int i=0; i<N_chains; i++) { nkc[i]->print_state(); } }
	TNKC<TParams, TLogger>* const get_chain(unsigned int index) { assert(index < N_chains); return nkc[index]; }
};

template<class TParams, class TLogger>
TParallelNKC<TParams, TLogger>::TParallelNKC(typename TNKC<TParams, TLogger >::pdf_t _pdf, typename TNKC<TParams, TLogger >::rand_state_t _rand_state, unsigned int _N, unsigned int _size, const double *const sigma, TParams& _params, TLogger& _logger, unsigned int _N_chains)
	: logger(_logger), params(_params), N(_N), nkc(NULL), component_stats(NULL), R(NULL), stats(_N)
{
	assert(_N_chains != 0);
	N_chains = _N_chains;
	nkc = new TNKC<TParams, TLogger>*[N_chains];
	component_stats = new TStats*[N_chains];
	for(unsigned int i=0; i<N_chains; i++) { nkc[i] = NULL; component_stats[i] = NULL; }
	#pragma omp parallel for
	for(unsigned int i=0; i<N_chains; i++) {
		nkc[i] = new TNKC<TParams, TLogger>(_pdf, _rand_state, N, _size, sigma, _params, _logger);
		component_stats[i] = &(nkc[i]->get_stats());
	}
	
	R = new double[N];
}

template<class TParams, class TLogger>
TParallelNKC<TParams, TLogger>::~TParallelNKC() {
	if(nkc != NULL) {
		for(unsigned int i=0; i<N_chains; i++) { if(nkc[i] != NULL) { delete nkc[i]; } }
		delete[] nkc;
	}
	if(component_stats != NULL) { delete[] component_stats; }
	if(R != NULL) { delete[] R; }
}

template<class TParams, class TLogger>
void TParallelNKC<TParams, TLogger>::step(unsigned int N_steps, bool log_step=true) {
	#pragma omp parallel shared(log_step) num_threads(N_chains)
	{
		unsigned int thread_ID = omp_get_thread_num();
		for(unsigned int i=0; i<N_steps; i++) { nkc[thread_ID]->step(log_step); }
		nkc[thread_ID]->flush();
		#pragma omp critical (append_stats)
		stats += nkc[thread_ID]->get_stats();
		
		#pragma omp barrier
	}
	Gelman_Rubin_diagnostic(component_stats, N_chains, R, N);
}

template<class TParams, class TLogger>
void TParallelNKC<TParams, TLogger>::burn_in(unsigned int N_rounds, unsigned int round_length, double target_acceptance=0.25, bool tune=true, bool adjust_proposal=true){
	#pragma omp parallel shared(N_rounds, round_length) num_threads(N_chains)
	{
		unsigned int thread_ID = omp_get_thread_num();
		nkc[thread_ID]->burn_in(N_rounds, round_length, target_acceptance, tune, adjust_proposal);
	}
}

template<class TParams, class TLogger>
void TParallelNKC<TParams, TLogger>::print_stats() {
	stats.print();
	std::cout << std::endl << "Gelman-Rubin diagnostic:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << (i==0 ? "" : "\t") << std::setprecision(5) << R[i]; }
	std::cout << std::endl;
	std::cout << "Acceptance rate: ";
	for(unsigned int i=0; i<N_chains; i++) { std::cout << std::setprecision(3) << 100.*get_chain(i)->get_acceptance_rate() << "%" << (i != N_chains-1 ? " " : ""); }
	std::cout << std::endl;
}


#endif