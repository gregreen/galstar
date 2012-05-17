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

void generate_test_data(double m[NBANDS], gsl_rng *rng, const TModel::Params &par, const double err[NBANDS])
{
	double mtrue[NBANDS], mext[NBANDS];
	for(unsigned int i=0; i<NBANDS; i++) {
		mtrue[i] = par.SED->v[i] + par.get_DM();
		mext[i]  = mtrue[i] + par.get_Ar() * TModel::Acoef[i];
		m[i] = mext[i] + gsl_ran_gaussian(rng, err[i]);
	}

	cerr << "# MOCK:    input: " << "Ar=" << par.get_Ar() << ", DM=" << par.get_DM() << ", Mr=" << par.get_Mr() << ", FeH=" << par.get_FeH() << "\n";
	cerr << "# MOCK:   m_true:"; for(unsigned int i=0; i<NBANDS; i++) { cerr << " " << mtrue[i]; }; cerr << "\n";
	cerr << "# MOCK:   m_ext :"; for(unsigned int i=0; i<NBANDS; i++) { cerr << " " <<  mext[i]; }; cerr << "\n";
	cerr << "# MOCK:   m_obs :"; for(unsigned int i=0; i<NBANDS; i++) { cerr << " " <<     m[i]; }; cerr << "\n";
}

bool construct_binners(TMultiBinner<4> &multibinner, vector<string> &output_fns, const vector<string> &output_pdfs, unsigned int N_bins=50) {
	#define G(name)         varname2int(name)
	static const regex e("([^:]+):([^,]+)(?:,([^,]+))?(?:,([^,]+))?");
	for(vector<string>::const_iterator i = output_pdfs.begin(); i != output_pdfs.end(); ++i) {
		ostringstream msg;
		msg << "# Outputting P(";
		
		// check overall format
		const std::string &pdfspec = *i;
		cmatch what;
		if(!regex_match(pdfspec.c_str(), what, e))
		{
			cerr << "Incorrect parameter format ('" << pdfspec << "')\n";
			return false;
		}
		
		// deduce size and check variables
		int ndim = 0;
		for(int i = 2; i != what.size() && what[i] != ""; ndim++, i++)
		{
			if(!(G(what[i])+1))
			{
				cerr << "Unrecognized model parameter '" << what[i] << "'\n";
				return false;
			}
			msg << (i > 2 ? ", " : "") << what[i];
		}
		msg << ")";
		
		// construct marginalizer. TODO: Add binners for dimensions other than 2
		output_fns.push_back(what[1]);
		switch(ndim)
		{
			//case 1:
			case 2:
				unsigned int bin_dim[2];
				unsigned int width[2];
				double min[2];
				double max[2];
				for(unsigned int k=0; k<ndim; k++) {
					bin_dim[k] = G(what[k+2]);
					width[k] = N_bins;//std_bin_width(what[k+2]);
					min[k] = std_bin_min(what[k+2]);
					max[k] = std_bin_max(what[k+2]);
				}
				multibinner.add_binner( new TBinner2D<4>(min, max, width, bin_dim) ); break;
			//case 3:
		}
		cerr << msg.str() << " into file " << what[1] << "\n";
	}
	#undef G
	return true;
}

// Output
int main(int argc, char **argv)
{
	vector<string> output_pdfs;
	string lf_fn = DATADIR "/MrLF.MSandRGB_v1.0.dat";
	string seds_fn = DATADIR "/MSandRGBcolors_v1.3.dat";
	string solar_pos = "8000 25";
	string par_thin = "2150 245";
	string par_thick = "0.13 3261 743";
	string par_halo = "0.0051 0.70 -2.62 27.8 -3.8";
	// TODO: Add in option to set metallicity parameters
	bool test = false;
	bool calcpdf = false;
	interval<double> Mr_range(ALL), FeH_range(ALL);
	range<double> DM_range(5,20,0.02), Ar_range(0,5,0.02);
	string datafn("NONE");
	string statsfn("NONE");
	string chainfn("NONE");
	bool brute_force = false;
	bool los = false;
	unsigned int giant_flag = 0;
	unsigned int N_steps = 1000;
	unsigned int N_samplers = 25;
	unsigned int N_bins = 50;
	unsigned int N_samples = 200;
	unsigned int N_threads = 4;
	
	// parse command line arguments
	namespace po = boost::program_options;
	po::options_description desc(std::string("Usage: ") + argv[0] + " <outfile1.txt:X1[,Y1]> [outfile2.txt:X2[,Y2] [...]]\n\nOptions");
	desc.add_options()
		("help", "produce this help message")
		("pdfs", po::value<vector<string> >(&output_pdfs)->multitoken(), "marginalized PDF to produce (can be given as the command line argument)")
		("lf", po::value<string>(&lf_fn), "luminosity function file")
		("seds", po::value<string>(&seds_fn), "SEDs file")
		("thindisk", po::value<string>(&par_thin), "Thin disk model parameters (l1 h1)")
		("thickdisk", po::value<string>(&par_thick), "Thick disk model parameters, (f_thin l2 h2)")
		("halo", po::value<string>(&par_halo), "Halo model parameters, (f_halo q n_inner R_br n_outer)")
		("test", po::value<bool>(&test)->zero_tokens(), "Assume the input contains (l b Ar DM Mr FeH uErr gErr rErr iErr zErr) and generate test data")
		("calcpdf", po::value<bool>(&calcpdf)->zero_tokens(), "Calculate log(p). Assume input contains l b DM Ar Mr FeH u g r i z uErr gErr rErr iErr zErr.")
		("range-M",   po::value<interval<double> >(&Mr_range),  "Range of absolute magnitudes to sample")
		("range-FeH", po::value<interval<double> >(&FeH_range), "Range of Fe/H to consider")
		("range-DM",   po::value<range<double> >(&DM_range), "DM grid to sample")
		("range-Ar",   po::value<range<double> >(&Ar_range), "Ar grid to sample")
		("datafile", po::value<string>(&datafn), "Stellar magnitudes and errors file")
		("statsfile", po::value<string>(&statsfn), "Base filename for statistics output")
		("chainfile", po::value<string>(&chainfn), "Base filename for chain output")
		("brute", "Use brute-force sampling")
		("los", "Sample entire line of sight at once")
		("steps", po::value<unsigned int>(&N_steps), "Minimum # of MCMC steps per sampler")
		("samplers", po::value<unsigned int>(&N_samplers), "# of affine samplers")
		("bins", po::value<unsigned int>(&N_bins), "# of bins along each axis")
		("samples", po::value<unsigned int>(&N_samples), "# of samples in each dimension for brute-force sampler")
		("threads", po::value<unsigned int>(&N_threads), "# of threads to run on")
		("dwarf", "Assume star is a dwarf (Mr > 4)")
		("giant", "Assume star is a giant (Mr < 4)")
	;
	po::positional_options_description pd;
	pd.add("pdfs", -1);
	
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);
	po::notify(vm);
	
	if (vm.count("help") || !output_pdfs.size()) { std::cout << desc << "\n"; return -1; }
	test = vm.count("test");
	if(vm.count("brute")) { brute_force = true; }
	if(vm.count("los")) { los = true; }
	if(vm.count("dwarf")) { giant_flag = 1; }
	if(vm.count("giant")) {
		if(giant_flag == 1) { std::cout << "--dwarf and --giant are incompatible options." << std::endl; return -1; }
		giant_flag = 2;
	}
	
	vector<string> output_fns;
	TMultiBinner<4> multibinner;
	if(!construct_binners(multibinner, output_fns, output_pdfs, N_bins)) { return -1; }
	
	////////////// Construct Model
	TModel model(lf_fn, seds_fn);
	#define PARSE(from, to) if(!(  istringstream(from) >> to  )) { std::cerr << "Error parsing " #from " (" << from << ")\n"; return -1; }
	PARSE(solar_pos,  model.R0 >> model.Z0);
	PARSE(par_thin,   model.L1 >> model.H1);
	PARSE(par_thick,  model.f  >> model.L2 >> model.H2);
	PARSE(par_halo,   model.fh >> model.qh  >> model.nh >> model.R_br2 >> model.nh_outer);
	model.fh_outer = model.fh * pow(1000.*model.R_br2/model.R0, model.nh-model.nh_outer);
	model.R_br2 = sqr(1000.*model.R_br2);
	#undef PARSE
	
	cerr << "# Galactic structure: " << model.R0 << " " << model.Z0 << " | " << model.L1 << " " << model.H1 << " | " << model.f << " " << model.L2 << " " << model.H2 << " | " << model.fh << " " << model.qh << " " << model.nh << "\n";
	
	
	// Construct data set //////////////////////////////////////////////////////////////////////////////////////////////////////////////
	TStellarData data;
	double m[NBANDS], err[NBANDS];
	double l, b;
	if(test) {
		////////////// Load Test Params and generate test data
		gsl_rng_env_setup();
		gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

		TModel::Params par;
		double Mr, FeH;
		// For debugging purposes: ///////////////////////
		/*l = 90.;
		b = 10.;
		par.Ar = 0.2;
		par.DM = 15.;
		Mr = 18.;
		FeH = -0.5;
		FOR(0, NBANDS) { err[i] = 0.05; }*/
		//////////////////////////////////////////////////
		cin >> l >> b >> par.Ar >> par.DM >> Mr >> FeH;
		for(unsigned int i=0; i<NBANDS; i++) { cin >> err[i]; }
		if(!cin) { cerr << "Error reading input data. Aborting.\n"; return -1; }
		
		par.SED = model.get_sed(Mr, FeH);
		generate_test_data(m, rng, par, err);
		
		TStellarData::TMagnitudes mag(m, err);
		data.star.push_back(mag);
		
		gsl_rng_free(rng);
	} else if(calcpdf) {
		double DM, Ar, Mr, FeH;
		cin >> l >> b >> DM >> Ar >> Mr >> FeH;
		for(unsigned int i=0; i<NBANDS; i++) { cin >> m[i]; }
		for(unsigned int i=0; i<NBANDS; i++) { cin >> err[i]; }
		TStellarData::TMagnitudes mag(m, err);
		print_logpdf(model, l, b, mag, data, m, err, DM, Ar, Mr, FeH);
		return 0;
	} else if(datafn != "NONE") {
		data.load_data(datafn);
		l = data.l;
		b = data.b;
	} else {
		////////////// Load Data
		cin >> l >> b;
		for(unsigned int i=0; i<NBANDS; i++) { cin >> m[i]; }
		for(unsigned int i=0; i<NBANDS; i++) { cin >> err[i]; }
		if(!cin) { cerr << "Error reading input data. Aborting.\n"; return -1; }
		TStellarData::TMagnitudes mag(m, err);
		data.star.push_back(mag);
	}
	
	model.DM_range = DM_range;
	model.Ar_range = Ar_range;
	model.Mr_range = Mr_range;
	model.FeH_range = FeH_range;
	cerr << "# Sampler:" 	<< " DM=[" << model.DM_range << "]" << " Ar=[" << model.Ar_range << "]"
				<< " Mr=[" << model.Mr_range << "]" << " FeH=[" << model.FeH_range << "]\n";
	
	
	// Run sampler ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if(los) {
		unsigned int N_stars = data.star.size();
		TStats stats(4*N_stars);
		bool converged;
		converged = sample_mcmc_los(model, l, b, *(data.star.begin()), data, multibinner, stats, N_steps, N_threads);
	} else {
		// Initialize class passed to sampling function, used to define the model and contain observed stellar magnitudes
		MCMCParams p(l, b, *data.star.begin(), model, data);
		p.giant_flag = giant_flag;	// Set flag which determines whether the model should consider only giants, only dwarfs, or both
		// Pass each star in turn to the sampling function
		unsigned int count = 0;
		unsigned int N_nonconverged = 0;
		for(vector<TStellarData::TMagnitudes>::iterator it = data.star.begin(); it != data.star.end(); ++it, ++count) {
			// Calculate posterior for current star
			std::cout << "=========================================" << std::endl;
			std::cout << "Calculating posterior for star #" << count << std::endl << std::endl;
			TStats stats(4);
			bool converged;
			if(brute_force) {
				stringstream outfn("");
				outfn << statsfn << "_" << count << ".chain";
				TChainLogger chainlogger(outfn.str(), 4, 1, true);
				converged = sample_brute_force(model, p, *it, multibinner, chainlogger, stats, N_samples, N_threads);
			} else {
				stringstream outfn("");
				if(chainfn != "NONE") {
					outfn << chainfn << "_" << count << ".chain";
				}
				if(giant_flag != 0) {
					converged = sample_affine(model, p, *it, multibinner, stats, outfn.str(), N_samplers, N_steps, N_threads);
				} else {
					converged = sample_affine_both(model, p, *it, multibinner, stats, outfn.str(), N_samplers, N_steps, N_threads);
				}
			}
			if(!converged) { N_nonconverged++; }
			// Write out the marginalized posteriors
			for(unsigned int i=0; i<multibinner.get_num_binners(); i++) {
				stringstream outfn("");
				if(test) {								// Determine filename
					outfn << output_fns.at(i) << ".txt";
				} else {
					outfn << output_fns.at(i) << "_" << count << ".txt";
				}
				multibinner.get_binner(i)->write_to_file(outfn.str());
			}
			// Write out summary of statistics
			if(statsfn != "NONE") {
				bool success = true;
				stringstream outfn("");
				if(test) {								// Determine filename
					outfn << statsfn << ".dat";
				} else {
					outfn << statsfn << "_" << count << ".dat";
				}
				std::fstream f;
				f.open(outfn.str().c_str(), std::ios::out | std::ios::binary);		// Write whether the fit converged as first byte
				f.write(reinterpret_cast<char*>(&converged), sizeof(converged));
				// Write max. likelihoods
				double ML[2];
				unsigned int N_binners = multibinner.get_num_binners();
				f.write(reinterpret_cast<char*>(&N_binners), sizeof(N_binners));	// Write # of max. likelihoods that follow
				for(unsigned int i=0; i<N_binners; i++) {
					TBinner2D<4> *binner = multibinner.get_binner(i);
					binner->get_ML(ML);
					for(unsigned int k=0; k<2; k++) {
						f.write(reinterpret_cast<char*>(&(binner->bin_dim[k])), sizeof(unsigned int));	// Write coordinate index
						f.write(reinterpret_cast<char*>(&(ML[k])), sizeof(double));			// Write position of max. likelihood for this coord.
					}
				}
				// Write means and covariance
				if(!f) {
					f.close();
					success = false;
				} else {
					f.close();
					success = stats.write_binary(outfn.str(), std::ios::app);
				}
				if(!success) { std::cerr << "# Could not write " << outfn.str() << std::endl; }
			}
			multibinner.clear();
		}
		
		std::cout << std::endl << "# Did not converge " << N_nonconverged << " times." << std::endl;
	}
	
	return 0;
}
