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

#include "binner.h"
#include "NKC.h"
#include "interpolater.h"
#include "smoothsort.h"
#include <astro/util.h>

static const int NBANDS = 5;

struct TSED
{
	double Mr, FeH;
	double v[NBANDS];	// Mu, Mg, Mr, Mi, Mz

	bool operator<(const TSED &b) const { return Mr < b.Mr || (Mr == b.Mr && FeH < b.FeH); }
};

struct TLF	// the luminosity function
{
	double Mr0, dMr;
	std::vector<double> lf;

	TLF(const std::string &fn) { load(fn); }

	double operator()(double Mr) const	// return the LF at position Mr (nearest neighbor interp.)
	{
		int idx = (int)floor((Mr - Mr0) / dMr + 0.5);
		if(idx < 0) { return lf.front(); }
		if(idx >= lf.size()) { return lf.back(); }
		return lf[idx];
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
	TLF lf;					// luminosity function
	TSED* seds;				// Stellar SEDs
	double dMr, dFeH, Mr_min, FeH_min;	// Sample spacing for stellar SEDs
	unsigned int N_FeH, N_Mr;
	
	static const double Acoef[NBANDS];	// Extinction coefficients relative to Ar
	
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
	
	TModel(const std::string& lf_, const std::string& seds_);
	~TModel() { delete seds; }
	
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
	
	bool load_data(std::string infile) {
		std::ifstream fin(infile.c_str());
		if(!fin.is_open()) {
			std::cout << "# Cannot open file " << infile << std::endl;
			return false;
		}
		std::cout << "# Loading stellar magnitudes from " << infile << " ..." << std::endl;
		fin >> l >> b;
		while(!fin.eof()) {
			TMagnitudes tmp;
			for(unsigned int i=0; i<NBANDS; i++) { fin >> tmp.m[i]; }
			for(unsigned int i=0; i<NBANDS; i++) { fin >> tmp.err[i]; tmp.err[i] = sqrt(tmp.err[i]); }
			star.push_back(tmp);
		}
		star.pop_back();
		fin.close();
		return true;
	}
};

struct MCMCParams {
	double l, b, cos_l, sin_l, cos_b, sin_b;
	TStellarData &data;			// Contains stellar magnitudes
	TModel &model;				// Contains galactic model information
	sorted_ptr_arr<TStellarData::TMagnitudes> sorted_stars;
	double DM_min, DM_max;			// Minimum and maximum distance moduli for which to precompute various priors
	#define DM_SAMPLES 10000
	
	// These two parameters are only used when fitting one star at a time
	double m[NBANDS];
	double err[NBANDS];
	
	TInterpolater *log_dn_arr, *f_halo_arr, *mu_disk_arr;
	
	MCMCParams(double _l, double _b, TStellarData::TMagnitudes &_mag, TModel &_model, TStellarData &_data) 
		: model(_model), data(_data), l(_l), b(_b), log_dn_arr(NULL), f_halo_arr(NULL), mu_disk_arr(NULL)
	{
		update(_mag);
		
		sorted_stars(_data.star);
		
		// Precompute trig functions
		cos_l = cos(0.0174532925*l);
		sin_l = sin(0.0174532925*l);
		cos_b = cos(0.0174532925*b);
		sin_b = sin(0.0174532925*b);
		
		// Precompute log(dn(DM)), f_halo(DM) and mu_disk(DM)
		DM_min = 0.01;
		DM_max = 25.;
		log_dn_arr = new TInterpolater(DM_SAMPLES, DM_min, DM_max);
		f_halo_arr = new TInterpolater(DM_SAMPLES, DM_min, DM_max);
		mu_disk_arr = new TInterpolater(DM_SAMPLES, DM_min, DM_max);
		double DM_i;
		for(unsigned int i=0; i<DM_SAMPLES; i++) {
			DM_i = log_dn_arr->get_x(i);
			(*log_dn_arr)[i] = model.log_dn(cos_l, sin_l, cos_b, sin_b, DM_i);
			(*f_halo_arr)[i] = model.f_halo(cos_l, sin_l, cos_b, sin_b, DM_i);
			(*mu_disk_arr)[i] = model.mu_disk(cos_l, sin_l, cos_b, sin_b, DM_i);
		}
	}
	
	~MCMCParams() {
		delete log_dn_arr;
		delete f_halo_arr;
		delete mu_disk_arr;
	}
	
	//inline unsigned int DM_index(double DM) { return (unsigned int)((DM-DM_min)/(DM_max-DM_min)*DM_SAMPLES + 0.5); }
	
	void update(TStellarData::TMagnitudes &mag) {
		for(unsigned int i=0; i<NBANDS; i++) {
			m[i] = mag.m[i];
			err[i] = mag.err[i];
		}
	}
	
	inline double log_dn_interp(const double DM) {
		if((DM < DM_min) || (DM > DM_max)) { return model.log_dn(cos_l, sin_l, cos_b, sin_b, DM); std::cout << "DM = " << DM << " !!!!" << std::endl; }
		//unsigned int index = DM_index(DM);
		return (*log_dn_arr)(DM);
	}
	
	inline double f_halo_interp(const double DM) {
		if((DM < DM_min) || (DM > DM_max)) { return model.f_halo(cos_l, sin_l, cos_b, sin_b, DM); std::cout << "DM = " << DM << " !!!!" << std::endl; }
		//unsigned int index = DM_index(DM);
		return (*f_halo_arr)(DM);
	}
	
	inline double mu_disk_interp(const double DM) {
		if((DM < DM_min) || (DM > DM_max)) { return model.mu_disk(cos_l, sin_l, cos_b, sin_b, DM); std::cout << "DM = " << DM << " !!!!" << std::endl; }
		//unsigned int index = DM_index(DM);
		return (*mu_disk_arr)(DM);
	}
	
	double log_p_FeH_fast(double DM, double FeH);
	
	#undef DM_SAMPLES
};

double calc_logP(const double *const x, unsigned int N, MCMCParams &p);

bool sample_mcmc(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TStellarData &data, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_steps, unsigned int N_threads);

bool sample_brute_force(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TStellarData &data, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_samples, unsigned int N_threads);

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
		return 5.;
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
