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
	void append(const TChain& chain, bool reweight=false, bool use_peak=true, double nsigma_max=1., double nsigma_peak=0.1, double chain_frac=0.05);	// Append a second chain to this one
	
	// Accessors
	unsigned int get_capacity() const;			// Return the capacity of the vectors used in the chain
	unsigned int get_length() const;			// Return the number of unique points in the chain
	double get_total_weight() const;			// Return the sum of the weights in the chain
	const double* get_element(unsigned int i) const;	// Return the i-th point in the chain
	double get_L(unsigned int i) const;			// Return the likelihood of the i-th point
	double get_w(unsigned int i) const;			// Return the weight of the i-th point
	double get_ln_Z_harmonic(bool use_peak=true, double nsigma_max=1., double nsigma_peak=0.1, double chain_frac=0.1) const;	// Estimate the Bayesian Evidence of the posterior using the Harmonic Mean Approximation
	void density_peak(double* const peak, double nsigma) const;	// Estimate coordinate with peak density
	
	// File IO
	bool save(std::string filename) const;				// Save the chain to file
	bool load(std::string filename, bool reserve_extra=false);	// Load the chain from file
	
	// Operators
	double* operator [](unsigned int i);		// Calls get_element
	void operator +=(const TChain& rhs);			// Calls append
	TChain& operator =(const TChain& rhs);		// Assignment operator
};


#endif // _CHAIN_H__