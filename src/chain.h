#ifndef _CHAIN_H__
#define CHAIN_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include <assert.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf_gamma.h>

#include <boost/cstdint.hpp>

#include "stats.h"

#ifndef PI
#define PI 3.14159265358979323
#endif

#ifndef SQRTPI
#define SQRTPI 1.77245385
#endif


// Gaussian mixture structure
//     Stores data necessary for representing a mixture of Gaussians, along
//     with workspaces for computing inverses.
struct TGaussianMixture {
	// Data
	unsigned int ndim, nclusters;
	double *w;
	double *mu;
	gsl_matrix **cov;
	gsl_matrix **inv_cov;
	gsl_matrix **sqrt_cov;
	double *det_cov;
	
	// Workspaces
	gsl_permutation *p;
	gsl_matrix *LU;
	gsl_eigen_symmv_workspace* esv;
	gsl_vector *eival;
	gsl_matrix *eivec;
	gsl_matrix *sqrt_eival;
	gsl_rng *r;
	
	// Constructor / Destructor
	TGaussianMixture(unsigned int _ndim, unsigned int _nclusters);
	~TGaussianMixture();
	
	// Accessors
	gsl_matrix* get_cov(unsigned int k);
	double get_w(unsigned int k);
	double* get_mu(unsigned int k);
	void draw(double *x);
	void print();
	
	// Mutators
	void invert_covariance();
	
	void density(const double *x, unsigned int N, double *res);
	double density(const double *x);
	
	void expectation_maximization(const double *x, const double *w, unsigned int N, unsigned int iterations=10);
};

/*************************************************************************
 *   Chain Class Prototype
 *************************************************************************/

class TChain {
private:
	std::vector<double> x;			// Elements in chain. Each point takes up N contiguous slots in array
	std::vector<double> L;			// Likelihood of each point in chain
	std::vector<double> w;			// Weight of each point in chain
	double total_weight;			// Sum of the weights
	unsigned int N, length, capacity;	// # of dimensions, length and capacity of chain
	
	std::vector<double> x_min;
	std::vector<double> x_max;

public:
	TStats stats;				// Keeps track of statistics of chain
	
	TChain(unsigned int _N, unsigned int _capacity);
	TChain(const TChain& c);
	TChain(std::string filename, bool reserve_extra=false);	// Construct the chain from a file
	~TChain();
	
	// Mutators
	void add_point(double* element, double L_i, double w_i);			// Add a point to the end of the chain
	void clear();									// Remove all the points from the chain
	void set_capacity(unsigned int _capacity);					// Set the capacity of the vectors used in the chain
	double append(const TChain& chain, bool reweight=false, bool use_peak=true, double nsigma_max=1., double nsigma_peak=0.1, double chain_frac=0.05, double threshold=1.e-5);	// Append a second chain to this one
	
	// Accessors
	unsigned int get_capacity() const;			// Return the capacity of the vectors used in the chain
	unsigned int get_length() const;			// Return the number of unique points in the chain
	double get_total_weight() const;			// Return the sum of the weights in the chain
	const double* get_element(unsigned int i) const;	// Return the i-th point in the chain
	double get_L(unsigned int i) const;			// Return the likelihood of the i-th point
	double get_w(unsigned int i) const;			// Return the weight of the i-th point
	
	// Computations on chain
	
	// Estimate the Bayesian Evidence of the posterior using the bounded Harmonic Mean Approximation
	double get_ln_Z_harmonic(bool use_peak=true, double nsigma_max=1., double nsigma_peak=0.1, double chain_frac=0.1) const;
	
	// Estimate coordinates with peak density by binning
	void density_peak(double* const peak, double nsigma) const;
	
	// Find a point in space with high density by picking a random point, drawing an ellipsoid, taking the mean coordinate within the ellipsoid, and then iterating
	void find_center(double* const center, gsl_matrix *const cov, gsl_matrix *const inv_cov, double* det_cov, double dmax=1., unsigned int iterations=5) const;
	
	void fit_gaussian_mixture(TGaussianMixture *gm, unsigned int iterations=10);
	
	// File IO
	bool save(std::string filename) const;				// Save the chain to file
	bool load(std::string filename, bool reserve_extra=false);	// Load the chain from file
	
	// Operators
	const double* operator [](unsigned int i);	// Calls get_element
	void operator +=(const TChain& rhs);		// Calls append
	TChain& operator =(const TChain& rhs);		// Assignment operator
};


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

// Sets inv_A to the inverse of A, and returns the determinant of A. If inv_A is NULL, then
// A is inverted in place. If worspaces p and LU are provided, the function does not have to
// allocate its own workspaces.
double invert_matrix(gsl_matrix* A, gsl_matrix* inv_A=NULL, gsl_permutation* p=NULL, gsl_matrix* LU=NULL);

// Find B s.t. B B^T = A. This is useful for generating vectors from a multivariate normal distribution.
void sqrt_matrix(gsl_matrix* A, gsl_matrix* sqrt_A=NULL, gsl_eigen_symmv_workspace* esv=NULL, gsl_vector *eival=NULL, gsl_matrix *eivec=NULL, gsl_matrix* sqrt_eival=NULL);

#endif // _CHAIN_H__