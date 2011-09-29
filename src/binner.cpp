
#include "binner.h"


TBinnerND::TBinnerND(double *_min, double *_max, unsigned int *width, unsigned int _N)
	: bin(NULL), min(NULL), max(NULL), dx(NULL), N(_N), index_workspace(NULL)
{
	min = new double[N];
	max = new double[N];
	dx = new double[N];
	index_workspace = new unsigned int[N];
	for(unsigned int i=0; i<N; i++) {
		min[i] = _min[i];
		max[i] = _max[i];
		dx[i] = (max[i]-min[i])/(double)width[i];
	}
	
	bin = new NDArray<double>(width, N);
	clear();
}

TBinnerND::~TBinnerND()
{
	delete[] min;
	delete[] max;
	delete[] dx;
	delete[] index_workspace;
	delete bin;
}

// Set the bins to zero
void TBinnerND::clear() {
	bin->fill(0.);
}

// Add data point to binner
void TBinnerND::add_point(double *pos, double weight) {
	// Check bounds
	for(unsigned int i=0; i<N; i++) {
		if((pos[i] < min[i]) || (pos[i] > max[i])) { return; }
	}
	// Add point
	for(unsigned int i=0; i<N; i++) { index_workspace[i] = (unsigned int)((pos[i] - min[i]) / dx[i]); }
	#pragma omp atomic
	bin->get_element(index_workspace, N) += weight;
}

// Normalize the bins either to the peak value, or to sum to unity
void TBinnerND::normalize(bool to_peak) {
	double norm = 0.;
	unsigned int N_bins = bin->get_size();
	if(to_peak) {	// Normalize to the peak probability
		for(unsigned int i=0; i<N_bins; i++) { if(bin->get_element(i) > norm) { norm = bin->get_element(i); } }
	} else {	// Normalize total probability to unity
		for(unsigned int i=0; i<N_bins; i++) { norm += bin->get_element(i); }
	}
	for(unsigned int i=0; i<N_bins; i++) { bin->get_element(i) /= norm; }
}

// Write the binned data to a binary file
void TBinnerND::write_to_file(std::string fname, bool ascii, bool log_pdf) {
	double log_bin_min = std::numeric_limits<double>::infinity();
	double tmp_bin;
	unsigned int *pos = new unsigned int[N];
	if(log_pdf) {
		NDArray<double>::iterator i_end = bin->end();
		for(NDArray<double>::iterator i=bin->begin(); i != i_end; ++i) {
			tmp_bin = *i;
			if((tmp_bin != 0.) && (tmp_bin < log_bin_min)) { log_bin_min = tmp_bin; }
		}
		log_bin_min = log(log_bin_min);
	}
	if(ascii) {
		std::ofstream outfile(fname.c_str());
		NDArray<double>::iterator i_end = bin->end();
		for(NDArray<double>::iterator i=bin->begin(); i != i_end; ++i) {
			tmp_bin = *i;
			if(log_pdf && (tmp_bin == 0.)) { tmp_bin = log_bin_min - 2.; } else { tmp_bin = log(tmp_bin); }
			i.get_pos(pos);
			for(unsigned int n=0; n<N; n++) {
				outfile << min[0]+dx[0]*((double)pos[n]+0.5) << "\t";
			}
			outfile << tmp_bin << "\n";
		}
		outfile.close();
	} else {
		std::fstream outfile(fname.c_str(), std::ios::binary | std::ios::out);
		for(unsigned int i=0; i<N; i++) {					// This second section gives the dimensions of the mesh
			unsigned int width_i = bin->get_width(i);
			outfile.write(reinterpret_cast<char *>(&N), sizeof(unsigned int));
			outfile.write(reinterpret_cast<char *>(&width_i), sizeof(unsigned int));
			outfile.write(reinterpret_cast<char *>(&min[i]), sizeof(double));
			outfile.write(reinterpret_cast<char *>(&max[i]), sizeof(double));
			outfile.write(reinterpret_cast<char *>(&dx[i]), sizeof(double));
		}
		NDArray<double>::iterator i_end = bin->end();
		for(NDArray<double>::iterator i=bin->begin(); i != i_end; ++i) {
			tmp_bin = *i;
			if(log_pdf && (tmp_bin == 0.)) { tmp_bin = log_bin_min - 2.; } else { tmp_bin = log(tmp_bin); }
			outfile.write(reinterpret_cast<char *>(&tmp_bin), sizeof(double));
		}
		outfile.close();
	}
	delete[] pos;
}
