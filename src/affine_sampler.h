
#ifndef _AFFINE_SAMPLER_H__
#define _AFFINE_SAMPLER_H__

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

#include "chain.h"
#include "stats.h"


/*************************************************************************
 *   Function Prototypes
 *************************************************************************/

void seed_gsl_rng(gsl_rng **r);

static void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R);


/*************************************************************************
 *   Affine Sampler class protoype
 *************************************************************************/

/* An affine-invariant ensemble sampler, introduced by Goodman & Weare (2010). */
template<class TParams, class TLogger>
class TAffineSampler {
	
	// Sampler settings
	unsigned int N;		// Dimensionality of parameter space
	unsigned int L;		// Number of component states in ensemble
	double sqrta;		// Square-root of dimensionless step scale a (a = 2 by default). Can be tuned to achieve desired acceptance rate.
	bool use_log;		// If true, <pdf> returns log(pi(X)). Else, <pdf> returns pi(X). Default value is <true>.
	
	// Current state
	struct TState;
	TState* X;		// Ensemble of states
	
	// Proposal states
	TState* Y;		// One proposal per state in ensemble
	bool* accept;		// Whether to accept this state
	
	TParams& params;	// Constant model parameters
	
	// Information about chain
	//TStats stats;		// Stores expectation values, covariance, etc.
	TChain chain;		// Contains the entire chain
	TLogger& logger;	// Object which logs states in the chain
	TState X_ML;		// Maximum likelihood point encountered
	boost::uint64_t N_accepted, N_rejected;		// # of steps which have been accepted and rejected. Used to tune and track acceptance rate.
	
	// Random number generator
	gsl_rng* r;
	
	// Private member functions
	void get_proposal(unsigned int j, double scale);	// Generate a proposal state for sampler j, with the given step scale
	
public:
	typedef double (*pdf_t)(const double *const _X, unsigned int _N, TParams& _params);
	typedef void (*rand_state_t)(double *const _X, unsigned int _N, gsl_rng* r, TParams& _params);
	
	// Constructor & destructor
	TAffineSampler(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, bool _use_log=true);
	~TAffineSampler();
	
	// Mutators
	void step(bool record_step=true);	// Advance each sampler in ensemble by one step
	void set_scale(double a);		// Set dimensionless step scale
	void flush(bool record_steps=true);	// Clear the weights in the ensemble and record the outstanding component states
	void clear();				// Clear the stats, acceptance information and weights
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return chain.stats; }
	TChain& get_chain() { return chain; }
	double get_scale() { return sqrta*sqrta; }
	double get_acceptance_rate() { return (double)N_accepted/(double)(N_accepted+N_rejected); }
	double get_Z_harmonic(double nsigma=1.) { return chain.get_Z_harmonic(nsigma); }
	void print_state();
	
private:
	rand_state_t rand_state;	// Function which generates a random state
	pdf_t pdf;			// pi(X), a function proportional to the target distribution
};


/*************************************************************************
 *   Parallel Affine Sampler Prototype
 *************************************************************************/

template<class TParams, class TLogger>
class TParallelAffineSampler {
	TAffineSampler<TParams, TLogger>** sampler;
	unsigned int N;
	unsigned int N_samplers;
	TStats stats;
	TStats** component_stats;
	TLogger& logger;
	TParams& params;
	double *R;
	
public:
	// Constructor & Destructor
	TParallelAffineSampler(typename TAffineSampler<TParams, TLogger>::pdf_t _pdf, typename TAffineSampler<TParams, TLogger>::rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, unsigned int _N_samplers, bool _use_log=true);
	~TParallelAffineSampler();
	
	// Mutators
	void step(unsigned int N_steps, bool record_steps, double cycle=0);					// Take the given number of steps in each affine sampler
	void set_scale(double a) { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->set_scale(a); } };	// Set the dimensionless step size a
	void clear() { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->clear(); }; stats.clear(); };
	
	// Accessors
	TLogger& get_logger() { return logger; }
	TParams& get_params() { return params; }
	TStats& get_stats() { return stats; }
	TStats& get_stats(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_stats(); }
	TChain get_chain();
	void get_GR_diagnostic(double *const GR) { for(unsigned int i=0; i<N; i++) { GR[i] = R[i]; } }
	double get_GR_diagnostic(unsigned int index) { return R[index]; }
	double get_scale(unsigned int index) { assert(index < N_samplers); return sampler[index]->get_scale(); }
	void print_stats();
	void print_state() { for(unsigned int i=0; i<N_samplers; i++) { sampler[i]->print_state(); } }
	TAffineSampler<TParams, TLogger>* const get_sampler(unsigned int index) { assert(index < N_samplers); return sampler[index]; }
};


/*************************************************************************
 *   Structs
 *************************************************************************/

// Component state type
template<class TParams, class TLogger>
struct TAffineSampler<TParams, TLogger>::TState {
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
 *   Affine Sampler Class Member Functions
 *************************************************************************/

/*************************************************************************
 *   Constructor and destructor
 *************************************************************************/

// Constructor
// Inputs:
// 	_pdf		Target distribution, up to a normalization constant
// 	_rand_state	Function which generates a random state, used for initialization of the chain
// 	_L		# of concurrent states in the ensemble
// 	_params		Misc. constant model parameters needed by _pdf
// 	_logger		Object which logs the chain in some way. It must have an operator()(double state[N], unsigned int weight).
// 			The logger could, for example, bin the chain, or just push back each state into a vector.
template<class TParams, class TLogger>
TAffineSampler<TParams, TLogger>::TAffineSampler(pdf_t _pdf, rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, bool _use_log)
	: pdf(_pdf), rand_state(_rand_state), params(_params), logger(_logger), N(_N), L(_L), X(NULL), Y(NULL), accept(NULL), r(NULL), use_log(_use_log), chain(_N, 1000*_L)
{
	// Seed the random number generator
	seed_gsl_rng(&r);
	
	// Generate the initial state and record the most likely point
	X = new TState[L];
	Y = new TState[L];
	accept = new bool[L];
	for(unsigned int i=0; i<L; i++) {
		X[i].initialize(N);
		Y[i].initialize(N);
	}
	unsigned int index_of_best = 0;
	for(unsigned int i=0; i<L; i++) {
		rand_state(X[i].element, N, r, params);
		X[i].pi = pdf(X[i].element, N, params);
		X[i].weight = 1;
		if(X[i] > X[index_of_best]) { index_of_best = i; }
	}
	X_ML = X[index_of_best];
	
	// Set the initial step scale. 2 is good for most situations.
	set_scale(2);
	
	// Initialize number of accepted and rejected steps to zero
	N_accepted = 0;
	N_rejected = 0;
}

// Destructor
template<class TParams, class TLogger>
TAffineSampler<TParams, TLogger>::~TAffineSampler() {
	gsl_rng_free(r);
	if(X != NULL) { delete[] X; X = NULL; }
	if(Y != NULL) { delete[] Y; Y = NULL; }
	if(accept != NULL) { delete[] accept; accept = NULL; }
}


/*************************************************************************
 *   Private functions
 *************************************************************************/

// Generate a proposal state
template<class TParams, class TLogger>
inline void TAffineSampler<TParams, TLogger>::get_proposal(unsigned int j, double scale) {
	// Choose a sampler to stretch from
	unsigned int k = gsl_rng_uniform_int(r, (long unsigned int)L - 1);
	if(k >= j) { k += 1; }
	// Determine the coordinates of the proposal
	for(unsigned int i=0; i<N; i++) {
		Y[j].element[i] = (1. - scale) * X[k].element[i] + scale * X[j].element[i];
	}
	// Get pdf(Y) and initialize weight of proposal point to unity
	Y[j].pi = pdf(Y[j].element, N, params);
	Y[j].weight = 1;
}


/*************************************************************************
 *   Mutators
 *************************************************************************/

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::step(bool record_step) {
	double scale, alpha, p;
	for(unsigned int j=0; j<L; j++) {
		// Determine the step scale and draw a proposal
		scale = (sqrta - 1./sqrta) * gsl_rng_uniform(r) + 1./sqrta;
		scale *= scale;
		get_proposal(j, scale);
		
		// Determine if the proposal is the maximum-likelihood point
		if(Y[j].pi > X_ML.pi) { X_ML = Y[j]; }
		
		// Determine whether to accept or reject
		accept[j] = false;
		if(use_log) {	// If <pdf> returns log probability
			// Determine the acceptance probability
			if(isinff(X[j].pi) && !(isinff(Y[j].pi))) {
				alpha = 1;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				alpha = (double)(N - 1) * log(scale) + Y[j].pi - X[j].pi;
			}
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] > -std::numeric_limits<double>::infinity())) {	// Accept if zero is rolled but proposal has nonzero probability
					accept[j] = true;
				} else if(log(p) < alpha) {
					accept[j] = true;
				}
			}
		} else {	// If <pdf> returns bare probability
			// Determine the acceptance probability
			if((X[j].pi == 0) && (Y[j].pi != 0)) {
				alpha = 2;	// Accept the proposal if the current state has zero probability and the proposed state doesn't
			} else {
				alpha = pow(scale, (double)(N - 1)) * Y[j].pi / X[j].pi;
			}
			// Decide whether to accept or reject
			if(alpha > 0.) {	// Accept if probability of acceptance is greater than unity
				accept[j] = true;
			} else {
				p = gsl_rng_uniform(r);
				if((p == 0.) && (Y[j] != 0)) {	// Accept if zero is rolled but proposal has nonzero probability
					accept[j] = true;
				} else if(p < alpha) {
					accept[j] = true;
				}
			}
		}
	}
	// Update ensemble
	for(unsigned int j=0; j<L; j++) {
		// Update sampler j
		if(accept[j]) {
			if(record_step) {
				//stats(X[j].element, X[j].weight);
				chain.add_point(X[j].element, X[j].pi, (double)(X[j].weight));
				#pragma omp critical (logger)
				logger(X[j].element, X[j].weight);
			}
			X[j] = Y[j];
			N_accepted++;
		} else {
			X[j].weight++;
			N_rejected++;
		}
	}
}

// Set the dimensionless step scale
template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::set_scale(double a) {
	sqrta = sqrt(a);
}

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::flush(bool record_steps) {
	for(unsigned int i=0; i<L; i++) {
		if(record_steps) {
			//stats(X[i].element, X[i].weight);
			chain.add_point(X[i].element, X[i].pi, (double)(X[i].weight));
			#pragma omp critical (logger)
			logger(X[i].element, X[i].weight);
		}
		X[i].weight = 0;
	}
}

// Clear the stats, acceptance information and weights
template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::clear() {
	for(unsigned int i=0; i<L; i++) {
		X[i].weight = 0;
	}
	//stats.clear();
	chain.clear();
	N_accepted = 0;
	N_rejected = 0;
}


/*************************************************************************
 *   Accessors
 *************************************************************************/

template<class TParams, class TLogger>
void TAffineSampler<TParams, TLogger>::print_state() {
	for(unsigned int i=0; i<L; i++) {
		std::cout << "p(X) = " << X[i].pi << std::endl;
		std::cout << "Weight = " << X[i].weight << std::endl << "X [" << i << "] = { ";
		for(unsigned int j=0; j<N; j++) { std::cout << (j == 0 ? "" : " ") << std::setprecision(3) << X[i].element[j]; }
		std::cout << " }" << std::endl << std::endl;
	}
}



/*************************************************************************
 *   Parallel Affine Sampler Class Member Functions
 *************************************************************************/

template<class TParams, class TLogger>
TParallelAffineSampler<TParams, TLogger>::TParallelAffineSampler(typename TAffineSampler<TParams, TLogger>::pdf_t _pdf, typename TAffineSampler<TParams, TLogger>::rand_state_t _rand_state, unsigned int _N, unsigned int _L, TParams& _params, TLogger& _logger, unsigned int _N_samplers, bool _use_log)
	: logger(_logger), params(_params), N(_N), sampler(NULL), component_stats(NULL), R(NULL), stats(_N)
{
	assert(_N_samplers > 1);
	N_samplers = _N_samplers;
	sampler = new TAffineSampler<TParams, TLogger>*[N_samplers];
	component_stats = new TStats*[N_samplers];
	for(unsigned int i=0; i<N_samplers; i++) { sampler[i] = NULL; component_stats[i] = NULL; }
	#pragma omp parallel for
	for(unsigned int i=0; i<N_samplers; i++) {
		sampler[i] = new TAffineSampler<TParams, TLogger>(_pdf, _rand_state, N, _L, _params, _logger, _use_log);
		component_stats[i] = &(sampler[i]->get_stats());
	}
	
	R = new double[N];
}

template<class TParams, class TLogger>
TParallelAffineSampler<TParams, TLogger>::~TParallelAffineSampler() {
	if(sampler != NULL) {
		for(unsigned int i=0; i<N_samplers; i++) { if(sampler[i] != NULL) { delete sampler[i]; } }
		delete[] sampler;
	}
	if(component_stats != NULL) { delete[] component_stats; }
	if(R != NULL) { delete[] R; }
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::step(unsigned int N_steps, bool record_steps, double cycle) {
	#pragma omp parallel shared(record_steps) num_threads(N_samplers)
	{
		unsigned int thread_ID = omp_get_thread_num();
		double base_a = sampler[thread_ID]->get_scale();
		for(unsigned int i=0; i<N_steps; i++) {
			if(cycle > 1) {
				if((i % 10) == 0) {
					sampler[thread_ID]->set_scale(base_a*cycle);
				} else if((i % 10) == 1) {
					sampler[thread_ID]->set_scale(base_a);
				}
			}
			sampler[thread_ID]->step(record_steps);
		}
		sampler[thread_ID]->flush(record_steps);
		#pragma omp critical (append_stats)
		stats += sampler[thread_ID]->get_stats();
		
		#pragma omp barrier
	}
	Gelman_Rubin_diagnostic(component_stats, N_samplers, R, N);
}

template<class TParams, class TLogger>
void TParallelAffineSampler<TParams, TLogger>::print_stats() {
	stats.print();
	std::cout << std::endl << "Gelman-Rubin diagnostic:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << (i==0 ? "" : "\t") << std::setprecision(5) << R[i]; }
	std::cout << std::endl;
	std::cout << "Acceptance rate: ";
	for(unsigned int i=0; i<N_samplers; i++) { std::cout << std::setprecision(3) << 100.*get_sampler(i)->get_acceptance_rate() << "%" << (i != N_samplers - 1 ? " " : ""); }
	std::cout << std::endl;
}

template<class TParams, class TLogger>
TChain TParallelAffineSampler<TParams, TLogger>::get_chain() {
	unsigned int capacity = 0;
	for(unsigned int i=0; i<N_samplers; i++) {
		capacity += sampler[i]->get_chain().get_length();
	}
	TChain tmp(N, capacity);
	for(unsigned int i=0; i<N_samplers; i++) {
		tmp += sampler[i]->get_chain();
	}
	return tmp;
}


/*************************************************************************
 *   Auxiliary Functions
 *************************************************************************/

#ifndef __SEED_GSL_RNG_
#define __SEED_GSL_RNG_
// Seed a gsl_rng with the Unix time in nanoseconds
inline void seed_gsl_rng(gsl_rng **r) {
	timespec t_seed;
	clock_gettime(CLOCK_REALTIME, &t_seed);
	long unsigned int seed = 1e9*(long unsigned int)t_seed.tv_sec;
	seed += t_seed.tv_nsec;
	*r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(*r, seed);
}
#endif


/*************************************************************************
 *   Null logger:
 * 	Fulfills the role of a logger for the affine sampler,
 * 	but doesn't actually log anything.
 *************************************************************************/

struct TNullLogger {
	void operator()(double* element, double weight) {}
};




#endif // _AFFINE_SAMPLER_H__