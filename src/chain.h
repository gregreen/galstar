#ifndef _CHAIN_H__
#define CHAIN_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>
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

public:
	TStats stats;				// Keeps track of statistics of chain
	
	TChain(unsigned int _N, unsigned int _capacity);
	TChain(std::string filename);			// Construct the chain from a file	// TODO
	~TChain();
	
	// Mutators
	void add_point(double* element, double L_i, double w_i);		// Add a point to the end of the chain
	void clear();								// Remove all the points from the chain
	void set_capacity(unsigned int _capacity);				// Set the capacity of the vectors used in the chain
	double* get_element(unsigned int i);					// Return the i-th point in the chain
	double get_Z_harmonic(double nsigma=1.);				// Estimate the Bayesian Evidence of the posterior using the Harmonic Mean Approximation
	void append(TChain& chain, bool reweight=false, double nsigma=1.);	// Append a second chain to this one
	
	// Accessors
	unsigned int get_capacity() const;		// Return the capacity of the vectors used in the chain
	unsigned int get_length() const;		// Return the number of unique points in the chain
	double get_total_weight() const;		// Return the sum of the weights in the chain
	double get_L(unsigned int i) const;		// Return the likelihood of the i-th point
	double get_w(unsigned int i) const;		// Return the weight of the i-th point
	
	// File IO
	bool save(std::string filename) const;		// Save the chain to file	// TODO
	bool load(std::string filename);		// Load the chain from file	// TODO
	
	// Operators
	double* operator [](unsigned int i);		// Calls get_element
	void operator +=(TChain& rhs);		// Calls append
};


#endif // _CHAIN_H__