#include "sampler.h"
#include "marginalize.h"
#include <sstream>
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
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
	FOR(0, NBANDS)
	{
		mtrue[i] = par.SED->v[i] + par.get_DM();
		mext[i]  = mtrue[i] + par.get_Ar() * TModel::Acoef[i];
		m[i]     = mext[i] + gsl_ran_gaussian(rng, err[i]);
	}

	cerr << "# MOCK:    input: " << "Ar=" << par.get_Ar() << ", DM=" << par.get_DM() << ", Mr=" << par.get_Mr() << ", FeH=" << par.get_FeH() << "\n";
	cerr << "# MOCK:   m_true:"; FOR(0, NBANDS) { cerr << " " << mtrue[i]; }; cerr << "\n";
	cerr << "# MOCK:   m_ext :"; FOR(0, NBANDS) { cerr << " " <<  mext[i]; }; cerr << "\n";
	cerr << "# MOCK:   m_obs :"; FOR(0, NBANDS) { cerr << " " <<     m[i]; }; cerr << "\n";
}

// Parse text such as output.txt:Mr,FeH,Ar and construct the requested marginalizer
bool construct_marginalizers(map<string, shared_ptr<TModel::Marginalizer> > &margs, const vector<string> &output_pdfs)
{
	#define G(name)         TModel::Params::varname2getter(name)
	static const regex e("([^:]+):([^,]+)(?:,([^,]+))?(?:,([^,]+))?");
	FOREACH(output_pdfs)
	{
		ostringstream msg;
		msg << "# Outputing P(";

		// check overal format
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
			if(!G(what[i]))
			{
				cerr << "Unrecognized model parameter '" << what[i] << "'\n";
				return false;
			}
			msg << (i > 2 ? ", " : "") << what[i];
		}
		msg << ")";

		// construct marginalizer
		string fn = what[1];
		switch(ndim)
		{
			case 1: margs[fn].reset( new Marginalizer<1>(G(what[2])) ); break;
			case 2: margs[fn].reset( new Marginalizer<2>(G(what[2]), G(what[3])) ); break;
			case 3: margs[fn].reset( new Marginalizer<3>(G(what[2]), G(what[3]), G(what[4])) ); break;
		}
		cerr << msg.str() << " into file " << fn << "\n";
	}
	#undef G
	return true;
}

// Output
int main(int argc, char **argv)
{
	vector<string> output_pdfs;
	string lf_fn = "MrLF.MSandRGB_v1.0.dat";
	string seds_fn = "MSandRGBcolors_v1.3.dat";
	string solar_pos = "8000 25";
	string par_thin = "2150 245";
	string par_thick = "0.13 3261 743";
	string par_halo = "0.0028 0.64 -2.77";
	bool test = false;
	interval<double> Mr_range(ALL), FeH_range(ALL);
	range<double> DM_range(5,20,0.02), Ar_range(0,1,0.02);

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
		("halo", po::value<string>(&par_halo), "Thick disk model parameters, (f_halo q n)")
		("test", po::value<bool>(&test)->zero_tokens()->implicit_value(true, "true"), "Assume the input contains (l b Ar DM Mr FeH uErr gErr rErr iErr zErr) and generate test data")
		("range-M",   po::value<interval<double> >(&Mr_range),  "Range of absolute magnitudes to sample")
		("range-FeH", po::value<interval<double> >(&FeH_range), "Range of Fe/H to consider")
		("range-DM",   po::value<range<double> >(&DM_range), "DM grid to sample")
		("range-Ar",   po::value<range<double> >(&Ar_range), "Ar grid to sample")
	;
	po::positional_options_description pd;
	pd.add("pdfs", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);
	po::notify(vm);

	if (vm.count("help") || !output_pdfs.size()) { std::cout << desc << "\n"; return -1; }


	map<string, shared_ptr<TModel::Marginalizer> > margs;
	if(!construct_marginalizers(margs, output_pdfs)) { return -1; }

	////////////// Construct Model
	TModel model(lf_fn, seds_fn);
	#define PARSE(from, to) if(!(  istringstream(from) >> to  )) { std::cerr << "Error parsing " #from " (" << from << ")\n"; return -1; }
	PARSE(solar_pos,  model.R0 >> model.Z0);
	PARSE(par_thin,   model.L1 >> model.H1);
	PARSE(par_thick,  model.f  >> model.L2 >> model.H2);
	PARSE(par_halo,   model.fh >> model.q  >> model.n);
	#undef PARSE

	cerr << "# Galactic structure: " << model.R0 << " " << model.Z0 << " | " << model.L1 << " " << model.H1 << " | " << model.f << " " << model.L2 << " " << model.H2 << " | " << model.fh << " " << model.q << " " << model.n << "\n";

	double m[NBANDS], err[NBANDS];
	double l, b;
	if(test)
	{
		////////////// Load Test Params and generate test data
		gsl_rng_env_setup();
		gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

		TModel::Params par;
		double Mr, FeH;
		cin >> l >> b >> par.Ar >> par.DM >> Mr >> FeH;
		FOR(0, NBANDS) { cin >> err[i]; }
		if(!cin) { cerr << "Error reading input data. Aborting.\n"; return -1; }

		par.SED = &*get_closest_SED(model.seds, Mr, FeH);
		generate_test_data(m, rng, par, err);

		gsl_rng_free(rng);
	}
	else
	{
		////////////// Load Data
		cin >> l >> b;
		FOR(0, NBANDS) { cin >> m[i]; }
		FOR(0, NBANDS) { cin >> err[i]; }
		if(!cin) { cerr << "Error reading input data. Aborting.\n"; return -1; }
	}

	////////////// Sample Posterior Distributions
	TModel::MarginalizerArray out;
	FOREACH(margs) { out.push_back(i->second.get()); }

	model.DM_range = DM_range;
	model.Ar_range = Ar_range;
	model.Mr_range = Mr_range;
	model.FeH_range = FeH_range;
	cerr << "# Sampler:" 	<< " DM=[" << model.DM_range << "]" << " Ar=[" << model.Ar_range << "]"
				<< " Mr=[" << model.Mr_range << "]" << " FeH=[" << model.FeH_range << "]\n";

	model.sample(out, l, b, m, err);

	////////////// Write out the results
	FOREACH(margs)
	{
		IMarginalizer &pdf = dynamic_cast<IMarginalizer&>(*(i->second));
		pdf.normalize_to_peak();

		ofstream out(i->first.c_str());
		pdf.output(out);
	}

	return 0;
}
