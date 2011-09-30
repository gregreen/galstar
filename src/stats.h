#ifndef _STATS_H__
#define _STATS_H__


#include <iostream>
#include <fstream>
#include <iomanip>
#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <vector>
#include <math.h>


class TStats {
	double *E_k;
	double *E_ij;
	unsigned int N;
	uint64_t N_items_tot;
	
public:
	// Constructor & Destructor
	TStats(unsigned int _N);
	~TStats();
	
	// Mutators
	void clear();							// Clear the contents of the statistics object
	void update(const double *const x, unsigned int weight);	// Update the chain from a an array of doubles with a weight
	void update(const TStats *const stats);
	
	void operator()(const double *const x, unsigned int weight);	// proxy for update()
	void operator()(const TStats *const stats);			// proxy for update()
	
	TStats& operator+=(const TStats &rhs);				// Add the data in another stats object to this one
	TStats& operator=(const TStats &rhs);				// Copy data from another stats object to this one, replacing existing data
	
	// Accessors
	double mean(unsigned int i) const;				// Return < x_i >
	double cov(unsigned int i, unsigned int j) const;		// Return covariance element Cov(i,j)
	uint64_t get_N_items() const;
	
	void print() const;						// Print out statistics
	bool write_binary(std::string fname, std::ios::openmode writemode = std::ios::out) const;	// Write statistics to binary file. Pass std::ios::app as writemode to append to end of existing file.
};

void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R, unsigned int N);

#endif // _STATS_H__