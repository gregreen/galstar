#ifndef sampler_h__
#define sampler_h__

#include <set>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <boost/cstdint.hpp>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sf_gamma.h>

#include "binner.h"
#include "chainlogger.h"
#include "NKC.h"
#include "affine_sampler.h"
#include "interpolation.h"
#include "smoothsort.h"
#include <astro/util.h>

static const int NBANDS = 5;

struct TSED
{
	double Mr, FeH;
	double v[NBANDS];	// Mu, Mg, Mr, Mi, Mz
	
	TSED() {
		for(unsigned int i=0; i<NBANDS; i++) { v[i] = 0; }
		Mr = 0; 
		FeH = 0;
	}
	
	bool operator<(const TSED &b) const { return Mr < b.Mr || (Mr == b.Mr && FeH < b.FeH); }
	
	// Operators required for bilinear interpolation
	
	TSED& operator=(const TSED &rhs) {
		for(unsigned int i=0; i<NBANDS; i++) { v[i] = rhs.v[i]; }
		Mr = rhs.Mr;
		FeH = rhs.FeH;
		return *this;
	}
	
	friend TSED operator+(const TSED &sed1, const TSED &sed2) {
		TSED tmp;
		for(unsigned int i=0; i<NBANDS; i++) { tmp.v[i] = sed1.v[i] + sed2.v[i]; }
		tmp.Mr = sed1.Mr + sed2.Mr;
		tmp.FeH = sed1.FeH + sed2.FeH;
		return tmp;
	}
	
	friend TSED operator-(const TSED &sed1, const TSED &sed2) {
		TSED tmp;
		for(unsigned int i=0; i<NBANDS; i++) { tmp.v[i] = sed1.v[i] - sed2.v[i]; }
		tmp.Mr = sed1.Mr - sed2.Mr;
		tmp.FeH = sed1.FeH - sed2.FeH;
		return tmp;
	}
	
	friend TSED operator*(const TSED &sed, const double &a) {
		TSED tmp;
		for(unsigned int i=0; i<NBANDS; i++) { tmp.v[i] = a*sed.v[i]; }
		tmp.Mr = a*sed.Mr;
		tmp.FeH = a*sed.FeH;
		return tmp;
	}
	
	friend TSED operator*(const double &a, const TSED &sed) {
		TSED tmp;
		for(unsigned int i=0; i<NBANDS; i++) { tmp.v[i] = a*sed.v[i]; }
		tmp.Mr = a*sed.Mr;
		tmp.FeH = a*sed.FeH;
		return tmp;
	}
	
	friend TSED operator/(const TSED &sed, const double &a) {
		TSED tmp;
		for(unsigned int i=0; i<NBANDS; i++) { tmp.v[i] = sed.v[i]/a; }
		tmp.Mr = sed.Mr / a;
		tmp.FeH = sed.FeH / a;
		return tmp;
	}
};

struct TLF	// the luminosity function
{
	double Mr0, dMr;
	std::vector<double> lf;
	TLinearInterp *lf_interp;

	TLF(const std::string &fn) : lf_interp(NULL) { load(fn); }
	~TLF() { delete lf_interp; }
	
	double operator()(double Mr) const	// return the LF at position Mr (linear interpolation)
	{
		return (*lf_interp)(Mr);
	}

	void load(const std::string &fn);
};

// samples P(DM,Ar,SED|m,l,b,GalStruct) ~
//	P(m|DM,Ar,SED,l,b,GalStruct) * P(DM,Ar,SED|l,b,GalStruct) =
//	P(m|DM,Ar,SED) * P(SED|DM,Ar,l,b,GalStruct) * P(DM,Ar|l,b,GalStruct) =
//	P(M|SED)       * P(SED|DM,l,b,GalStruct) * P(DM|Ar,l,b,GalStruct) * P(Ar|l,b,Galstruct) =
//	P(M|SED)       * P(SED|DM,l,b,GalStruct) * P(DM|l,b,Galstruct)    * P(Ar) =
//	where M=m-DM-A(Ar) and:
//		P(M|SED) is the likelihood of SED given measurement M,
//		P(SED|DM,l,b,GalStruct) is proportional to the luminosity function,
//		P(DM|l,b,Galstruct) is proportional to the number of stars
//			at (DM,l,b) and
//		P(Ar) is the prior for Ar
struct TModel
{
	// Model parameters: galactic ctr. distance, solar offset, disk scale height & length
	double     R0, Z0;			// Solar position
	double     H1, L1;			// Thin disk
	double f,  H2, L2;			// Galactic structure (thin and thick disk)
	double fh,  qh,  nh, R_br2, nh_outer;	// Galactic structure (power-law halo)
	double fh_outer;
	TLF lf;							// luminosity function
	TSED* seds;						// Stellar SEDs
	double dMr, dFeH, Mr_min, FeH_min, Mr_max, FeH_max;	// Sample spacing for stellar SEDs
	unsigned int N_FeH, N_Mr;
	
	double Acoef[NBANDS];	// Extinction coefficients relative to Ar
	
	TBilinearInterp<TSED> *sed_interp;	// Bilinear interpolation of stellar SEDs in Mr and FeH
	
	struct Params				// Parameter space that can be sampled
	{
		double DM, Ar;
		const TSED *SED;
		
		double get_DM()  const { return DM; }
		double get_Ar()  const { return Ar; }
		double get_Mr()  const { return SED->Mr; }
		double get_FeH() const { return SED->FeH; }
		
		typedef double (TModel::Params::*Getter)() const;
		static Getter varname2getter(const std::string &var);
	};
	
	// Parameter ranges over which to sample
	peyton::util::range<double>      DM_range, Ar_range;
	peyton::util::interval<double>   Mr_range, FeH_range;
	
	TModel(const std::string& lf_fn, const std::string& seds_fn, const double (&Acoef_)[NBANDS]);
	~TModel() { delete seds; delete sed_interp; }
	
	void computeCartesianPositions(double &X, double &Y, double &Z, double cos_l, double sin_l, double cos_b, double sin_b, double d) const;
	double rho_disk(double R, double Z) const;
	double rho_halo(double R, double Z) const;
	double log_dn(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const;
	double log_p_FeH(double cos_l, double sin_l, double cos_b, double sin_b, double DM, double FeH) const;	// From Ivezich et al. 2008
	double f_halo(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const;
	double mu_disk(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const;
	
	TSED* get_sed(const double Mr, const double FeH) const;
	void get_sed_inline(TSED** sed_out, const double Mr, const double FeH) const;
	unsigned int sed_index(const double Mr, const double FeH) const;
};


// MCMC Stuff ////////////////////////////////////////////////////////////////////////////////////////////////
#define _DM 0
#define _Ar 1
#define _Mr 2
#define _FeH 3

struct TStellarData {
	struct TMagnitudes {
		double m[NBANDS];
		double err[NBANDS];
		double DM_est;		// Estimate of distance to star. Used in sorting stars.
		
		TMagnitudes() {}
		
		TMagnitudes(double (&_m)[NBANDS], double (&_err)[NBANDS]) {
			for(unsigned int i=0; i<NBANDS; i++) {
				m[i] = _m[i];
				err[i] = _err[i];
			}
			DM_est = 0.;
		}
		
		bool operator>(TMagnitudes& rhs) { return (rhs.DM_est > DM_est); }
		
		TMagnitudes& operator=(const TMagnitudes& rhs) {
			for(unsigned int i=0; i<NBANDS; i++) {
				m[i] = rhs.m[i];
				err[i] = rhs.err[i];
			}
			DM_est = rhs.DM_est;
			return *this;
		}
	};
	
	double l, b;
	std::vector<TMagnitudes> star;
	
	TStellarData(std::string infile) { load_data(infile); }
	TStellarData() {}
	
	TMagnitudes& operator[](const unsigned int &index) { return star.at(index); }
	
	// Load magnitudes and errors of stars along one line of sight, along with (l,b) for the given l.o.s.
	bool load_data(std::string infile, double err_floor=0.001) {
		std::ifstream fin(infile.c_str());
		if(!fin.is_open()) {
			std::cout << "# Cannot open file " << infile << std::endl;
			return false;
		}
		std::cout << "# Loading stellar magnitudes from " << infile << " ..." << std::endl;
		fin >> l >> b;
		while(!fin.eof()) {
			TMagnitudes tmp;
			double err_tmp;
			for(unsigned int i=0; i<NBANDS; i++) { fin >> tmp.m[i]; }
			for(unsigned int i=0; i<NBANDS; i++) { fin >> err_tmp; tmp.err[i] = sqrt(err_tmp*err_tmp + err_floor*err_floor); }
			star.push_back(tmp);
		}
		star.pop_back();
		fin.close();
		return true;
	}
	
	// Load magnitudes and errors of stars along one line of sight, along with (l,b) for the given l.o.s. Same as load_data, but for binary files.
	// Expected format:
	// 	Header:
	// 		l		(double)
	// 		b		(double)
	// 		N_stars		(uint32)
	// 	Data - For each star:
	// 		mag[NBANDS]	(double)
	// 		err[NBANDS]	(double)
	bool load_data_binary(std::string infile, double err_floor=0.001) {
		std::fstream f(infile.c_str(), std::ios::in | std::ios::binary);
		if(!f) { f.close(); return false; }
		
		// Read in header
		uint32_t N_stars;
		f.read(reinterpret_cast<char*>(&l), sizeof(l));
		f.read(reinterpret_cast<char*>(&b), sizeof(b));
		f.read(reinterpret_cast<char*>(&N_stars), sizeof(N_stars));
		
		// Read in each star
		star.reserve(N_stars);
		for(uint32_t i=0; i<N_stars; i++) {
			TMagnitudes tmp;
			f.read(reinterpret_cast<char*>(&(tmp.m[0])), NBANDS*sizeof(double));
			f.read(reinterpret_cast<char*>(&(tmp.err[0])), NBANDS*sizeof(double));
			for(unsigned int i=0; i<NBANDS; i++) { tmp.err[i] = sqrt(tmp.err[i]*tmp.err[i] + err_floor*err_floor); }
			star.push_back(tmp);
		}
		
		if(f.fail()) { f.close(); return false; }
		
		f.close();
		return true;
	}
};

struct MCMCParams {
	double l, b, cos_l, sin_l, cos_b, sin_b;
	TStellarData &data;			// Contains stellar magnitudes
	TModel &model;				// Contains galactic model information
	double DM_min, DM_max;			// Minimum and maximum distance moduli for which to precompute various priors
	#define DM_SAMPLES 10000
	
	// These two parameters are only used when fitting one star at a time
	double m[NBANDS];
	double err[NBANDS];
	
	// Flag whether to include only giants, only dwarfs, or both
	unsigned int giant_flag;	// 0 = both, 1 = dwarfs, 2 = giants
	
	TLinearInterp *log_dn_arr, *f_halo_arr, *mu_disk_arr;
	
	MCMCParams(double _l, double _b, TStellarData::TMagnitudes &_mag, TModel &_model, TStellarData &_data)
		: model(_model), data(_data), l(_l), b(_b), log_dn_arr(NULL), f_halo_arr(NULL), mu_disk_arr(NULL)
	{
		update(_mag);
		
		// Precompute trig functions
		cos_l = cos(0.0174532925*l);
		sin_l = sin(0.0174532925*l);
		cos_b = cos(0.0174532925*b);
		sin_b = sin(0.0174532925*b);
		
		// Precompute log(dn(DM)), f_halo(DM) and mu_disk(DM)
		DM_min = 0.01;
		DM_max = 25.;
		log_dn_arr = new TLinearInterp(DM_min, DM_max, DM_SAMPLES);
		f_halo_arr = new TLinearInterp(DM_min, DM_max, DM_SAMPLES);
		mu_disk_arr = new TLinearInterp(DM_min, DM_max, DM_SAMPLES);
		double DM_i;
		for(unsigned int i=0; i<DM_SAMPLES; i++) {
			DM_i = log_dn_arr->get_x(i);
			(*log_dn_arr)[i] = model.log_dn(cos_l, sin_l, cos_b, sin_b, DM_i);
			(*f_halo_arr)[i] = model.f_halo(cos_l, sin_l, cos_b, sin_b, DM_i);
			(*mu_disk_arr)[i] = model.mu_disk(cos_l, sin_l, cos_b, sin_b, DM_i);
		}
		
		// Set the giant flag to include both giants and dwarfs
		giant_flag = 0;
	}
	
	~MCMCParams() {
		delete log_dn_arr;
		delete f_halo_arr;
		delete mu_disk_arr;
	}
	
	void update(TStellarData::TMagnitudes &mag) {
		for(unsigned int i=0; i<NBANDS; i++) {
			m[i] = mag.m[i];
			err[i] = mag.err[i];
		}
	}
	
	inline double log_dn_interp(const double DM) {
		if((DM < DM_min) || (DM > DM_max)) { return model.log_dn(cos_l, sin_l, cos_b, sin_b, DM); std::cout << "DM = " << DM << " !!!!" << std::endl; }
		return (*log_dn_arr)(DM);
	}
	
	inline double f_halo_interp(const double DM) {
		if((DM < DM_min) || (DM > DM_max)) { return model.f_halo(cos_l, sin_l, cos_b, sin_b, DM); std::cout << "DM = " << DM << " !!!!" << std::endl; }
		return (*f_halo_arr)(DM);
	}
	
	inline double mu_disk_interp(const double DM) {
		if((DM < DM_min) || (DM > DM_max)) { return model.mu_disk(cos_l, sin_l, cos_b, sin_b, DM); std::cout << "DM = " << DM << " !!!!" << std::endl; }
		return (*mu_disk_arr)(DM);
	}
	
	double log_p_FeH_fast(double DM, double FeH);
	
	#undef DM_SAMPLES
};

// Functions for line-of-sight calculations
void ran_state_los(double *const x_0, unsigned int N, gsl_rng *r, MCMCParams &p);
double calc_logP_los(const double *const x, unsigned int N, MCMCParams &p);
double log_prior(const double *const x, unsigned int N, MCMCParams &p);
double log_permutation_likelihood(const double *const x, unsigned int N, MCMCParams &p, gsl_permutation *data_order);
bool sample_mcmc_los(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TStellarData &data, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_steps, unsigned int N_threads);

// Functions for individual star
void ran_state(double *const x_0, unsigned int N, gsl_rng *r, MCMCParams &p);
double calc_logP(const double *const x, unsigned int N, MCMCParams &p);
bool sample_mcmc(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_samplers, unsigned int N_steps, unsigned int N_threads);
bool sample_affine(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, std::string chain_out, unsigned int N_samplers, unsigned int N_steps, unsigned int N_threads);
bool sample_affine_both(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, std::string chain_out, unsigned int N_samplers, unsigned int N_steps, unsigned int N_threads);
bool sample_brute_force(TModel &model,MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TChainLogger &chainlogger, TStats &stats, unsigned int N_samples, unsigned int N_threads);

// Debugging functions
void print_logpdf(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TStellarData &data, double (&m)[5], double (&err)[5], double DM, double Ar, double Mr, double FeH);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////


inline int varname2int(const std::string &varname) {
	if(varname == "DM") {
		return _DM;
	} else if(varname == "Ar") {
		return _Ar;
	} else if(varname == "Mr") {
		return _Mr;
	} else if(varname == "FeH") {
		return _FeH;
	}
	return -1;
}

inline double std_bin_min(const std::string &varname) {
	if(varname == "DM") {
		return 5.;
	} else if(varname == "Ar") {
		return 0.;
	} else if(varname == "Mr") {
		return -1.;
	} else if(varname == "FeH") {
		return -2.5;
	}
	return -1.;
}

inline double std_bin_max(const std::string &varname) {
	if(varname == "DM") {
		return 20.;
	} else if(varname == "Ar") {
		return 10.;
	} else if(varname == "Mr") {
		return 28.;
	} else if(varname == "FeH") {
		return 0.;
	}
	return -1.;
}

inline double std_bin_min(unsigned int i) {
	if(i == _DM) {
		return 5.;
	} else if(i == _Ar) {
		return 0.;
	} else if(i == _Mr) {
		return -1.;
	} else if(i == _FeH) {
		return -2.5;
	}
	return -1.;
}

inline double std_bin_max(unsigned int i) {
	if(i == _DM) {
		return 20.;
	} else if(i == _Ar) {
		return 10.;
	} else if(i == _Mr) {
		return 28.;
	} else if(i == _FeH) {
		return 0.;
	}
	return -1.;
}


#endif // sampler_h__
