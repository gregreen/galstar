#include "galstar_config.h"

#include "sampler.h"

#include <sstream>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/regex.hpp>
#include <boost/program_options.hpp>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include <astro/useall.h>

using namespace boost;
using namespace std;


// Construct binners from commandline input
bool construct_binners(TMultiBinner<4> &multibinner, vector<string> &output_fns, const vector<string> &output_pdfs) {
	#define G(name)         varname2int(name)
	
	// Accepts input of the form "DM_Ar.dat:DM[5,20,100],Ar[0,10,200]", where "DM_Ar.dat" is the output filename,
	// "DM[5,20,100]" indicates that DM should be binned from 5 to 20, with 100 bins, and "Ar[0,10,100]" is intepreted analogously.
	// The following regular expression would parse our example input as
	// 1: DM_Ar.dat
	// 2: DM
	// 3: 5
	// 4: 20
	// 5: 100
	// 6: Ar
	// 7: 0
	// 8: 10
	// 9: 200
	static const regex e("([^:]+):([^,^\\[^\\]]+)\\[(-?\\d+|-?\\d+.\\d),(-?\\d+|-?\\d+.\\d),(\\d+)\\],([^,^\\(^\\)]+)\\[(-?\\d+|-?\\d+.\\d),(-?\\d+|-?\\d+.\\d),(\\d+)\\]");
	
	// Loop though each binner specification
	for(vector<string>::const_iterator i = output_pdfs.begin(); i != output_pdfs.end(); ++i) {
		// Check that input format matches input
		const std::string &pdfspec = *i;
		cmatch what;
		if(!regex_match(pdfspec.c_str(), what, e)) {
			cerr << "Incorrect parameter format ('" << pdfspec << "')\n";
			return false;
		}
		
		// Store the filename for this binner
		output_fns.push_back(what[1]);
		
		// Construct the binner
		unsigned int bin_dim[2];
		unsigned int width[2];
		double min[2];
		double max[2];
		for(unsigned int k=0; k<2; k++) {
			if(!G(what[4*k+2]+1)) {		// Check that the variable name is valid
				cerr << "Unrecognized model parameter: '" << what[4*k+2] << "'" << endl;
				return false;
			}
			bin_dim[k] = G(what[4*k+2].str().c_str());
			min[k] = atof(what[4*k+3].str().c_str());
			max[k] = atof(what[4*k+4].str().c_str());
			if(atoi(what[4*k+5].str().c_str()) < 0) {	// Check that the # of bins specified is nonzero
				cerr << "Negative # of bins specified for model parameter '" << what[4*k+2] << "'" << endl;
				return false;
			}
			width[k] = atoi(what[4*k+5].str().c_str());
		}
		multibinner.add_binner( new TBinner2D<4>(min, max, width, bin_dim) );
		
		cerr << "# Outputting P(" << what[2] << ", " << what[6] << ") into file " << what[1] << endl;
	}
	
	#undef G
	return true;
}

bool generate_test_photometry(string test_fn, TModel &model, TStellarData &data) {
	ifstream f(test_fn.c_str());
	if(!f) { f.close(); return false; }
	
	string line;
	double DM, Ar, Mr, FeH;
	gsl_rng *r;
	seed_gsl_rng(&r);
	
	unsigned int count = 0;
	
	while(getline(f, line)) {
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment
		
		istringstream ss(line);
		if(count == 0) {
			ss >> data.l >> data.b;
			cout << "(l, b) = " << data.l << " " << data.b << endl;
		} else {
			ss >> DM >> Ar >> Mr >> FeH;
			TStellarData::TMagnitudes tmp;
			TSED sed = (*model.sed_interp)(Mr, FeH);
			cout << "stellar parameters: " << DM << " " << Ar << " " << Mr << " " << FeH << endl;
			cout << "grizy = {";
			for(unsigned int i=0; i<NBANDS; i++) {
				tmp.err[i] = 0.02;				// TODO: Set realistic errors from spline
				tmp.m[i] = sed.v[i] + DM + Ar*model.Acoef[i] + gsl_ran_gaussian_ziggurat(r, tmp.err[i]);
				cout << (i != 0 ? " " : "") << tmp.m[i];
			}
			cout << "}" << endl;
			data.star.push_back(tmp);
		}
		
		count++;
	}
	
	cout << "Generated test photometry using stellar parameters in " << test_fn << "." << endl;
	
	gsl_rng_free(r);
	f.close();
	return true;
}

int main(int argc, char **argv) {
	vector<string> output_pdfs;
	string lf_fn = DATADIR "/PSMrLF.dat";		// For SDSS, use "/MrLF.MSandRGB_v1.0.dat";
	string seds_fn = DATADIR "/PScolors.dat";	// For SDSS, use "/MSandRGBcolors_v1.3.dat"
	string solar_pos = "8000 25";
	string par_thin = "2150 245";
	string par_thick = "0.13 3261 743";
	string par_halo = "0.0051 0.70 -2.62 27.8 -3.8";
	// TODO: Add in option to set metallicity parameters
	vector<string> infile_str;
	uint32_t pix_index;
	string infile("NONE");
	string statsfn("NONE");
	string test_fn = "NONE";
	string photometry = "PS";
	double errfloor = 20.;
	bool sparse = true;
	bool append = false;
	bool noprior = false;
	unsigned int giant_flag = 0;
	unsigned int N_steps = 2000;
	unsigned int N_samplers = 40;
	unsigned int N_samples = 200;
	unsigned int N_threads = 4;
	
	const double AcoefSDSS[NBANDS] = {4.239/2.285, 3.303/2.285, 2.285/2.285, 1.698/2.285, 1.263/2.285};
	const double AcoefPS1[NBANDS] = {3.172/2.271, 2.271/2.271, 1.682/2.271, 1.322/2.271, 1.087/2.271};
	
	// parse command line arguments
	namespace po = boost::program_options;
	po::options_description desc(std::string("Usage: ") + argv[0] + " <outfile1.dat:X1[min,max,bins],Y1[min,max,bins]> [outfile2.dat:X2[min,max,bins],Y2[min,max,bins] [...]]\n\nOptions");
	desc.add_options()
		("help", "produce this help message")
		("pdfs", po::value<vector<string> >(&output_pdfs)->multitoken(), "marginalized PDF to produce (can be given as the command line argument)")
		("lf", po::value<string>(&lf_fn), "luminosity function file")
		("seds", po::value<string>(&seds_fn), "SEDs file")
		("photometry", po::value<string>(&photometry), "Photometric system ('PS' or 'SDSS', default 'PS')")
		("errfloor", po::value<double>(&errfloor), "Error to add in quadrature (in millimags)")
		("thindisk", po::value<string>(&par_thin), "Thin disk model parameters (l1 h1)")
		("thickdisk", po::value<string>(&par_thick), "Thick disk model parameters, (f_thin l2 h2)")
		("halo", po::value<string>(&par_halo), "Halo model parameters, (f_halo q n_inner R_br n_outer)")
		("infile", po::value<vector<string> >(&infile_str)->multitoken(), "Stellar magnitudes and errors file, and index of pixel in file.")
		("statsfile", po::value<string>(&statsfn), "Base filename for statistics output")
		("steps", po::value<unsigned int>(&N_steps), "Minimum # of MCMC steps per sampler")
		("samplers", po::value<unsigned int>(&N_samplers), "# of affine samplers")
		("samples", po::value<unsigned int>(&N_samples), "# of samples in each dimension for brute-force sampler")
		("threads", po::value<unsigned int>(&N_threads), "# of threads to run on")
		("nonsparse", "Write binned PDFs as full arrays (produces significantly larger output).")
		("append", "Append output to existing files.")
		("dwarf", "Assume star is a dwarf (Mr > 4)")
		("giant", "Assume star is a giant (Mr < 4)")
		("test", po::value<string>(&test_fn), "ASCII containing stellar parameters with which to generate test input.")
		("noprior", "Do not apply priors (i.e. only calculate likelihood).")
	;
	po::positional_options_description pd;
	pd.add("pdfs", -1);
	
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);
	po::notify(vm);
	
	if(vm.count("help") || !output_pdfs.size()) { cout << desc << endl; return -1; }
	if(vm.count("dwarf")) { giant_flag = 1; }
	if(vm.count("giant")) {
		if(giant_flag == 1) { cout << "--dwarf and --giant are incompatible options." << endl; return -1; }
		giant_flag = 2;
	}
	if(infile_str.size() == 2) {
		infile = infile_str[0];
		pix_index = (uint32_t)atoi(infile_str[1].c_str());
	} else if(test_fn == "NONE") {
		cerr << "If 'infile' is not specified, 'test' must be given." << endl << "E.g. '--infile pix.in 10' selects the 10th set of stars from pix.in." << endl;
		return -1;
	}
	if(vm.count("nonsparse")) { sparse = false; }
	if(vm.count("append")) { append = true; }
	if(vm.count("noprior")) { noprior = true; }
	errfloor /= 1000.;
	
	vector<string> output_fns;
	TMultiBinner<4> multibinner;
	if(!construct_binners(multibinner, output_fns, output_pdfs)) { return -1; }
	
	// Construct Model ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	double Acoef_tmp[NBANDS];
	if(photometry == "PS") {
		for(unsigned int i=0; i<NBANDS; i++){ Acoef_tmp[i] = AcoefPS1[i]; }
	} else if(photometry == "SDSS") {
		for(unsigned int i=0; i<NBANDS; i++){ Acoef_tmp[i] = AcoefSDSS[i]; }
	} else {
		cerr << "Invalid photometry: '" << photometry << "'. Valid options are 'PS' and 'SDSS'." << endl;
		return -1;
	}
	TModel model(lf_fn, seds_fn, Acoef_tmp);
	#define PARSE(from, to) if(!(  istringstream(from) >> to  )) { std::cerr << "Error parsing " #from " (" << from << ")\n"; return -1; }
	PARSE(solar_pos,  model.R0 >> model.Z0);
	PARSE(par_thin,   model.L1 >> model.H1);
	PARSE(par_thick,  model.f  >> model.L2 >> model.H2);
	PARSE(par_halo,   model.fh >> model.qh  >> model.nh >> model.R_br2 >> model.nh_outer);
	model.fh_outer = model.fh * pow(1000.*model.R_br2/model.R0, model.nh-model.nh_outer);
	model.R_br2 = sqr(1000.*model.R_br2);
	#undef PARSE
	
	cerr << "# Galactic structure: " << model.R0 << " " << model.Z0 << " | " << model.L1 << " " << model.H1 << " | " << model.f << " " << model.L2 << " " << model.H2 << " | " << model.fh << " " << model.qh << " " << model.nh << "\n";
	
	// Load stellar photometry ////////////////////////////////////////////////////////////////////////////////////////////////////////
	TStellarData data;
	if(test_fn == "NONE") {
		if(!data.load_data_binary(infile, pix_index, errfloor)) {
			cerr << "Failed to load data from " << infile << "." << endl;
			return -1;
		}
	} else { // Generate photometry from stellar model
		if(!generate_test_photometry(test_fn, model, data)) {
			cerr << "Failed to generate test photometry from " << test_fn << "." << endl;
			return -1;
		}
	}
	double m[NBANDS], err[NBANDS];
	double l, b;
	l = data.l;
	b = data.b;
	
	// Run sampler ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// Initialize class passed to sampling function, used to define the model and contain observed stellar magnitudes
	MCMCParams p(l, b, *data.star.begin(), model, data);
	p.giant_flag = giant_flag;	// Set flag which determines whether the model should consider only giants, only dwarfs, or both
	p.noprior = noprior;
	
	// Pass each star in turn to the sampling function
	unsigned int count = 0;
	unsigned int N_nonconverged = 0;
	unsigned int N_stars = data.star.size();
	for(vector<TStellarData::TMagnitudes>::iterator it = data.star.begin(); it != data.star.end(); ++it, ++count) {
		// Calculate posterior for current star
		cout << "=========================================" << endl;
		cout << "Calculating posterior for star #" << count+1 << " of " << N_stars << endl << endl;
		for(unsigned int i=0; i<NBANDS; i++) {
			cout << "m[" << i << "] = " << it->m[i] << " +- " << it->err[i] << endl;
		}
		TStats stats(4);
		bool converged;
		double evidence;
		if(giant_flag != 0) {
			converged = sample_affine(model, p, *it, multibinner, stats, evidence, N_samplers, N_steps, N_threads);
		} else {
			converged = sample_affine_both(model, p, *it, multibinner, stats, evidence, N_samplers, N_steps, N_threads);
		}
		if(!converged) { N_nonconverged++; }
		
		// Output binned pdfs and statistics for current star
		bool append_to_file = (append || (count != 0));
		// Write out the marginalized posteriors
		for(unsigned int i=0; i<multibinner.get_num_binners(); i++) {
			bool write_success = multibinner.get_binner(i)->write_binary(output_fns.at(i), append_to_file, sparse);
		}
		// Write out summary of statistics
		if(statsfn != "NONE") {
			bool write_success = stats.write_binary(statsfn, converged, evidence, append_to_file);
		}
		multibinner.clear();
	}
	
	cerr << endl << "# Failed to converge " << N_nonconverged << " time(s)." << endl << endl;
	
	return 0;
}
