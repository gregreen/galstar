#ifndef _CHAINLOGGER__
#define _CHAINLOGGER__

#include <iostream>
#include <string>
#include <fstream>

// Log the chain element-for-element
struct TChainLogger {
	std::string filename;
	std::ofstream outfile;
	unsigned int N_elements, dim, N_samplers;
	bool output_weight;
	
	TChainLogger(std::string _filename, unsigned int _dim, unsigned int _N_samplers, bool _output_weight=false)
		: filename(_filename), dim(_dim), N_samplers(_N_samplers), outfile(_filename.c_str()), output_weight(_output_weight)
	{
		//outfile.write(reinterpret_cast<char*>(&dim), sizeof(dim));
		//outfile.write(reinterpret_cast<char*>(&N_samplers), sizeof(N_samplers));
		outfile << dim << std::endl;
		outfile << N_samplers << std::endl;
		N_elements = 0;
	}
	
	~TChainLogger() {
		outfile.close();
	}
	
	void add_point(const double *X, unsigned int N, double weight) {
		//outfile.write(reinterpret_cast<const char*>(X), dim*sizeof(double));
		for(unsigned int i = 0; i < dim; i++) { outfile << X[i] << "\t"; }
		if(output_weight) { outfile << weight; }
		outfile << std::endl;
		N_elements++;
	}
	
	void operator ()(const double *const pos, double weight) { add_point(pos, dim, weight); }
};


#endif // _CHAINLOGGER__