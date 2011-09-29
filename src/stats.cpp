
#include "stats.h"

TStats::TStats(unsigned int _N)
	: E_k(NULL), E_ij(NULL), N(_N)
{
	E_k = new double[N];
	E_ij = new double[N*N];
	clear();
}

TStats::~TStats() {
	delete[] E_k;
	delete[] E_ij;
}


// Clear all the contents of the statistics object
void TStats::clear() {
	for(unsigned int i=0; i<N; i++) {
		E_k[i] = 0.;
		for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] = 0.; }
	}
	N_items_tot = 0;
}

// Update the chain from a an array of doubles with a weight
void TStats::update(const double *const x, unsigned int weight) {
	for(unsigned int i=0; i<N; i++) {
		E_k[i] += x[i] * (double)weight;
		for(unsigned int j=i; j<N; j++) {
			E_ij[i+N*j] += x[i] * x[j] * (double)weight;
			E_ij[N*i+j] = E_ij[i+N*j];
		}
	}
	N_items_tot += (uint64_t)weight;
}

// Update the chain from the statistics in another TStats object
void TStats::update(const TStats *const stats) {
	assert(stats->N == N);
	for(unsigned int i=0; i<N; i++) {
		E_k[i] += stats->E_k[i];
		for(unsigned int j=i; j<N; j++) {
			E_ij[i+N*j] += stats->E_ij[i+N*j];
			E_ij[N*i+j] = E_ij[i+N*j];
		}
	}
	N_items_tot += stats->N_items_tot;
}

// Update the chain from the statistics in another TStats object
void TStats::operator()(const TStats *const stats) { update(stats); }

// Update the chain from a an array of doubles with a weight
void TStats::operator()(const double *const x, unsigned int weight) { update(x, weight); }

// Add the data in another stats object to this one
TStats& TStats::operator+=(const TStats &rhs) {
	assert(rhs.N == N);
	N_items_tot += rhs.N_items_tot;
	for(unsigned int i=0; i<N; i++) {
		E_k[i] += rhs.E_k[i];
		for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] += rhs.E_ij[i+N*j]; }
	}
	return *this;
}

// Copy data from another stats object to this one, replacing existing data
TStats& TStats::operator=(const TStats &rhs) {
	assert(rhs.N == N);
	if(&rhs != this) {
		N_items_tot = rhs.N_items_tot;
		for(unsigned int i=0; i<N; i++) {
			E_k[i] = rhs.E_k[i];
			for(unsigned int j=0; j<N; j++) { E_ij[i+N*j] = rhs.E_ij[i+N*j]; }
		}
	}
	return *this;
}

// Return covariance element Cov(i,j)
double TStats::cov(unsigned int i, unsigned int j) const { return (E_ij[i+N*j] - E_k[i]*E_k[j]/(double)N_items_tot)/(double)N_items_tot; }

// Return < x_i >
double TStats::mean(unsigned int i) const { return E_k[i] / (double)N_items_tot; }

uint64_t TStats::get_N_items() const { return N_items_tot; }

// Print out statistics
void TStats::print() const {
	std::cout << "Mean:" << std::endl;
	for(unsigned int i=0; i<N; i++) { std::cout << "\t" << std::setprecision(3) << mean(i) << "\t+-\t" << sqrt(cov(i, i)) << std::endl; }
	std::cout << std::endl;
	std::cout << "Covariance:" << std::endl;
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=0; j<N; j++) { std::cout << "\t" << cov(i, j); }
		std::cout << std::endl;
	}
}

// Write statistics to binary file. Pass std::ios::app as writemode to append to end of existing file.
bool TStats::write_binary(std::string fname, std::ios::openmode writemode) const {
	std::fstream f;
	f.open(fname.c_str(), writemode | std::ios::out | std::ios::binary);
	if(!f) { f.close(); return false; }	// Return false if the file could not be opened
	// Write number of dimensions
	unsigned int dim = N;
	f.write(reinterpret_cast<char*>(&dim), sizeof(dim));
	// Write mean values
	double tmp;
	for(unsigned int i=0; i<N; i++) {
		tmp = mean(i);
		f.write(reinterpret_cast<char*>(&tmp), sizeof(tmp));
	}
	// Write upper triangle (including diagonal) of covariance matrix
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=i; j<N; j++) {
			tmp = cov(i, j);
			f.write(reinterpret_cast<char*>(&tmp), sizeof(tmp));
		}
	}
	// Return false if there was a write error, else true
	if(!f) { f.close(); return false; }
	f.close();
	return true;
}

void Gelman_Rubin_diagnostic(TStats **stats_arr, unsigned int N_chains, double *R, unsigned int N) {
	// Run some basic checks on the input to ensure that G-R statistics can be calculated
	assert(N_chains > 1);	// More than one chain
	unsigned int N_items_tot = stats_arr[0]->get_N_items();
	for(unsigned int i=1; i<N_chains; i++) { assert(stats_arr[i]->get_N_items() == N_items_tot); }	// Each chain is of the same length
	
	std::vector<double> W(N, 0.);		// Mean within-chain variance
	std::vector<double> B(N, 0.);		// Between-chain variance
	std::vector<double> Theta(N, 0.);	// Mean of means (overall mean)
	
	// Calculate mean within chain variance and overall mean
	for(unsigned int i=0; i<N_chains; i++) {
		for(unsigned int k=0; k<N; k++) {
			W[k] += stats_arr[i]->cov(k,k);
			Theta[k] += stats_arr[i]->mean(k);
		}
	}
	for(unsigned int k=0; k<N; k++) {
		W[k] /= (double)N_chains;
		Theta[k] /= (double)N_chains;
	}
	
	// Calculate variance between chains
	double tmp;
	for(unsigned int i=0; i<N_chains; i++) {
		for(unsigned int k=0; k<N; k++) {
			tmp = stats_arr[i]->mean(k) - Theta[k];
			B[k] += tmp*tmp;
		}
	}
	for(unsigned int k=0; k<N; k++) { B[k] /= (double)N_chains - 1.; }
	
	// Calculate estimated variance
	for(unsigned int k=0; k<N; k++) { R[k] = 1. - 1./(double)N_items_tot + B[k]/W[k]; }
}
