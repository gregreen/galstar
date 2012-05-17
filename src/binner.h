#ifndef _BINNER_H__
#define _BINNER_H__

//#include "npy.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <limits>
#include <vector>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <boost/shared_ptr.hpp>

#include "ndarray.h"


/** **************************************************************************************************************************************************************************
 * ND Binner
 *****************************************************************************************************************************************************************************/

// TODO: Write accessors and marginalizers
class TBinnerND {
	// Data
	NDArray<double> *bin;
	double *min, *max, *dx;
	unsigned int N;
	
	unsigned int *index_workspace;
	
public:
	// Constructor & Destructor
	TBinnerND(double *_min, double *_max, unsigned int *width, unsigned int _N);
	~TBinnerND();
	
	// Mutators //////////////////////////////////////////////////////////////////////////////////////////////
	void add_point(double *pos, double weight);	// Add data point to binner
	void clear();					// Set the bins to zero
	void normalize(bool to_peak=true);		// Normalize the bins either to the peak value, or to sum to unity
	
	void operator ()(double *pos, double weight) { add_point(pos, weight); }
	
	// Accessors /////////////////////////////////////////////////////////////////////////////////////////////
	void write_to_file(std::string fname, bool ascii=true, bool log_pdf=false);	// Write the binned data to a binary file
	void print_bins();								// Print out the bins to cout
};


/** **************************************************************************************************************************************************************************
 * 2D Binner
 *****************************************************************************************************************************************************************************/

// Fast and simple 2D data binner
template<unsigned int N>
struct TBinner2D {
	// Data
	double min[2];
	double max[2];
	double dx[2];
	unsigned int width[2];
	unsigned int bin_dim[2];
	double **bin;
	
	uint64_t **nsamples;
	
	// Constructor & Destructor
	TBinner2D(double (&_min)[2], double (&_max)[2], unsigned int (&_width)[2], unsigned int (&_bin_dim)[2]);
	~TBinner2D();
	
	// Mutators //////////////////////////////////////////////////////////////////////////////////////////////
	void add_point(double (&pos)[N], double weight);	// Add data point to binner
	void add_point(const double *const pos, double weight);
	void clear();						// Set the bins to zero
	void normalize(bool to_peak=true);			// Normalize the bins either to the peak value, or to sum to unity
	
	void operator ()(double (&pos)[N], double weight) { add_point(pos, weight); }
	void operator ()(const double *const pos, double weight) { add_point(pos, weight); }
	
	// Accessors /////////////////////////////////////////////////////////////////////////////////////////////
	void write_to_file(std::string fname, bool ascii=true, bool log_pdf=false, double zero_diff=-10.);	// Write binned data to file
	void load_from_file(std::string fname, bool ascii=true, bool log_pdf=false, double zero_diff=-10.);	// Load binned data from file
	void write_npz(std::string fname);									// Write numpy .npz file
	void print_bins();											// Print out the bins to cout
	void get_ML(double (&ML)[2]);										// Return maximum likelihood
};


template<unsigned int N>
TBinner2D<N>::TBinner2D(double (&_min)[2], double (&_max)[2], unsigned int (&_width)[2], unsigned int (&_bin_dim)[2]) {
	for(unsigned int i=0; i<2; i++) {
		min[i] = _min[i];
		max[i] = _max[i];
		width[i] = _width[i];
		assert(_bin_dim[i] < N);
		bin_dim[i] = _bin_dim[i];
		dx[i] = (max[i] - min[i]) / (double)width[i];
	}
	bin = new double*[width[0]];
	nsamples = new uint64_t*[width[0]];
	for(unsigned int i=0; i<width[0]; i++) {
		bin[i] = new double[width[1]];
		nsamples[i] = new uint64_t[width[1]];
		//for(unsigned int k=0; k<width[1]; k++) { bin[i][k] = 0.; }
	}
	clear();
}

template<unsigned int N>
TBinner2D<N>::~TBinner2D() {
	for(unsigned int i=0; i<width[0]; i++) { delete[] bin[i]; delete[] nsamples[i]; }
	delete[] bin; delete[] nsamples;
}

// Set the bins to zero
template<unsigned int N>
void TBinner2D<N>::clear() {
	for(unsigned int i=0; i<width[0]; i++) {
		for(unsigned int k=0; k<width[1]; k++) { bin[i][k] = 0.; nsamples[i][k] = 0; }
	}
}

// Add data point to binner
template<unsigned int N>
void TBinner2D<N>::add_point(double (&pos)[N], double weight) {
	// Check bounds
	for(unsigned int i=0; i<2; i++) {
		if((pos[bin_dim[i]] < min[i]) || (pos[bin_dim[i]] > max[i])) { return; }
	}
	// Add point
	unsigned int index[2];
	for(unsigned int i=0; i<2; i++) {
		index[i] = (unsigned int)((pos[bin_dim[i]] - min[i]) / dx[i]);
		//#pragma omp critical(cout)
		//std::cout << (pos[bin_dim[i]] - min[i]) / dx[i] << "\t" << index[i] << std::endl;
	}
	#pragma omp atomic
	bin[index[0]][index[1]] += weight;
	#pragma omp atomic
	nsamples[index[0]][index[1]] += 1;
}

template<unsigned int N>
void TBinner2D<N>::add_point(const double *const pos, double weight) {
	// Check bounds
	for(unsigned int i=0; i<2; i++) {
		if((pos[bin_dim[i]] < min[i]) || (pos[bin_dim[i]] > max[i])) { return; }
	}
	// Add point
	unsigned int index[2];
	for(unsigned int i=0; i<2; i++) { index[i] = (unsigned int)((pos[bin_dim[i]] - min[i]) / dx[i]); }
	#pragma omp atomic
	bin[index[0]][index[1]] += weight;
}

// Normalize the bins either to the peak value, or to sum to unity
template<unsigned int N>
void TBinner2D<N>::normalize(bool to_peak) {
	double norm = 0.;
	if(to_peak) {	// Normalize to the peak probability
		for(unsigned int j=0; j<width[0]; j++) {
			for(unsigned int k=0; k<width[1]; k++) { if(bin[j][k] > norm) { norm = bin[j][k]; } }
		}
	} else {	// Normalize total probability to unity
		for(unsigned int j=0; j<width[0]; j++) {
			for(unsigned int k=0; k<width[1]; k++) { norm += bin[j][k]; }
		}
	}
	for(unsigned int j=0; j<width[0]; j++) {
		for(unsigned int k=0; k<width[1]; k++) { bin[j][k] /= norm; }
	}
}

// Sets ML to the coordinate with maximum likelihood
template<unsigned int N>
void TBinner2D<N>::get_ML(double (&ML)[2]) {
	double max = -1.;
	for(unsigned int j=0; j<width[0]; j++) {
		for(unsigned int k=0; k<width[1]; k++) {
			if(bin[j][k] > max) {
				max = bin[j][k];
				ML[0] = min[0] + ((double)j + 0.5) * dx[0];
				ML[1] = min[1] + ((double)k + 0.5) * dx[1];
			}	
		}
	}
}

// Write the binned data to a binary file
template<unsigned int N>
void TBinner2D<N>::write_to_file(std::string fname, bool ascii, bool log_pdf, double zero_diff) {
	double log_bin_min = std::numeric_limits<double>::infinity();
	double tmp_bin;
	if(log_pdf) {
		for(unsigned int j=0; j<width[0]; j++) {
			for(unsigned int k=0; k<width[1]; k++) { if((bin[j][k] != 0.) && (bin[j][k] < log_bin_min)) { log_bin_min = bin[j][k]; } }
		}
		log_bin_min = log(log_bin_min);
	}
	if(ascii) {	// Write ASCII
		std::ofstream outfile(fname.c_str());
		for(unsigned int j=0; j<width[0]; j++) {
			for(unsigned int k=0; k<width[1]; k++) {
				if(!log_pdf) { tmp_bin = bin[j][k]; } else if(bin[j][k] == 0.) { tmp_bin = log_bin_min + zero_diff; } else { tmp_bin = log(bin[j][k]); }
				outfile << min[0]+dx[0]*((double)j+0.5) << "\t" << min[1]+dx[1]*((double)k+0.5) << "\t" << tmp_bin << "\n";
			}
		}
		outfile.close();
	} else {	// Write binary
		std::fstream outfile(fname.c_str(), std::ios::binary | std::ios::out);
		unsigned int tmp;
		for(unsigned int i=0; i<2; i++) {					// This second section gives the dimensions of the mesh
			tmp = width[i];
			outfile.write(reinterpret_cast<char *>(&tmp), sizeof(unsigned int));
			outfile.write(reinterpret_cast<char *>(&min[i]), sizeof(double));
			outfile.write(reinterpret_cast<char *>(&max[i]), sizeof(double));
			outfile.write(reinterpret_cast<char *>(&dx[i]), sizeof(double));
		}
		for(unsigned int j=0; j<width[0]; j++) {
			if(log_pdf) {
				for(unsigned int k=0; k<width[1]; k++) {
					//if(!log_pdf) { tmp_bin = bin[j][k]; } else if(bin[j][k] == 0.) { tmp_bin = log_bin_min + zero_diff; } else { tmp_bin = log(bin[j][k]); }
					//outfile.write(reinterpret_cast<char *>(&tmp_bin), sizeof(double));
					if(bin[j][k] == 0.) { tmp_bin = log_bin_min + zero_diff; } else { tmp_bin = log(bin[j][k]); }
					outfile.write(reinterpret_cast<char *>(&tmp_bin), sizeof(double));
				}
			} else {
				outfile.write(reinterpret_cast<char *>(bin[j]), width[0]*sizeof(double));
			}
		}
		outfile.close();
	}
}


// Write the binned data to a numpy .npz file
template<unsigned int N>
void TBinner2D<N>::write_npz(std::string fname) {
	// Flatten the bin array
	double *tmp_bin = new double[width[0]*width[1]];
	for(unsigned int i=0; i<width[0]; i++) {
		memcpy(&(tmp_bin[i*width[0]]), bin[i], width[0]*sizeof(bin[i][0]));
	}
	// Write the bin array to npz
	int fortran_order = 0;
	//npy_save_double(fname, fortran_order, 2, width, tmp_bin);
}


template<unsigned int N>
void TBinner2D<N>::print_bins() {
	for(int k=(int)width[1]-1; k>=0; k--) {
		std::cout << std::setprecision(3) << min[1]+dx[1]*((double)k+0.5) << "\t||\t";
		for(unsigned int j=0; j<width[0]; j++) { std::cout << std::setprecision(3) << bin[j][k] << "\t"; }
		//for(unsigned int j=0; j<width[0]; j++) { std::cout << nsamples[j][k] << "\t"; }
		std::cout << std::endl;
	}
	for(unsigned int j=0; j<width[0]+2; j++) { std::cout << "====\t"; }
	std::cout << std::endl << "\t||\t";
	for(unsigned int j=0; j<width[0]; j++) { std::cout << std::setprecision(3) << min[0]+dx[0]*((double)j+0.5) << "\t"; }
	std::cout << std::endl;
}


/** **************************************************************************************************************************************************************************
 * Container for multiple 2D Binners
 *****************************************************************************************************************************************************************************/

template<unsigned int N>
class TMultiBinner {
	std::vector< boost::shared_ptr<TBinner2D<N> > > binner_arr;
	
public:
	TMultiBinner() {}
	~TMultiBinner() {}
	
	void operator ()(double (&pos)[N], double weight) {
		for(typename std::vector< boost::shared_ptr<TBinner2D<N> > >::iterator it = binner_arr.begin(); it != binner_arr.end(); ++it) { (**it)(pos, weight); }
	}
	
	void operator ()(const double *const pos, double weight) {
		for(typename std::vector< boost::shared_ptr<TBinner2D<N> > >::iterator it = binner_arr.begin(); it != binner_arr.end(); ++it) { (**it)(pos, weight); }
	}
	
	void clear() {
		for(typename std::vector< boost::shared_ptr<TBinner2D<N> > >::iterator it = binner_arr.begin(); it != binner_arr.end(); ++it) { (*it)->clear(); }
	}
	
	void add_binner(boost::shared_ptr<TBinner2D<N> > binner) { binner_arr.push_back(binner); }
	
	// WARNING: Pass only TBinner2D<N> pointers that have been created with the <new> command. Otherwise, a segfault occurs when the TMultiBinner<N> object is destroyed.
	void add_binner(TBinner2D<N> *binner) {
		boost::shared_ptr<TBinner2D<N> > binner_ptr(binner);
		binner_arr.push_back(binner_ptr);
	}
	
	TBinner2D<N> *get_binner(unsigned int i) {
		assert(i<binner_arr.size());
		return &(*binner_arr.at(i));
	}
	
	unsigned int get_num_binners() { return binner_arr.size(); }
};

#endif
