#ifndef sampler_h__
#define sampler_h__

#include <set>
#include <vector>
#include <string>
#include <cmath>

#include <astro/util.h>

static const int NBANDS = 5;

struct TSED
{
	double Mr, FeH;
	double v[NBANDS];	// Mu, Mg, Mr, Mi, Mz

	bool operator<(const TSED &b) const { return Mr < b.Mr || (Mr == b.Mr && FeH < b.FeH); }
};
typedef std::vector<TSED> TSEDs;	// set of SEDs
void load_seds(TSEDs &seds, const std::string &fn);	// load SEDs from a file
TSEDs::iterator get_closest_SED(TSEDs& seds, double Mr, double FeH);

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
	double     R0, Z0;		// Solar position
	double     H1, L1;		// Thin disk
	double f,  H2, L2;		// Galactic structure (thin and thick disk)
	double fh,  q,  n;		// Galactic structure (power-law halo)
	TLF lf;				// luminosity function
	TSEDs seds;			// Stellar SEDs

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

	struct MarginalizerStack;
	struct Marginalizer
	{
		virtual void sampling_begin(int nthreads) = 0;
		virtual void sampling_end() = 0;

		virtual void operator()(const Params &p, double logL, int threadId) = 0;
		virtual ~Marginalizer() {}
	};
	struct MarginalizerArray : public Marginalizer, public std::vector<Marginalizer*>
	{
		virtual void sampling_begin(int nthreads) { FOREACH(*this) { (*i)->sampling_begin(nthreads); } }
		virtual void sampling_end()               { FOREACH(*this) { (*i)->sampling_end(); }           }

		virtual void operator()(const Params &p, double logL, int threadId)
		{
			FOREACH(*this) { (**i)(p, logL, threadId); }
		}
	};

	TModel(const std::string& lf_, const std::string& seds_);

	void computeCartesianPositions(double &X, double &Y, double &Z, double ldeg, double bdeg, double d) const;
	double expdisk(double X, double Y, double Z) const;
	double dn(double l, double b, double DM) const;

	// evenly sample the (DM, Ar, SEDs) space, and pass the posterior
	// probability to the output object
	void sample(TModel::Marginalizer& out, double l, double b, const double m[5], const double err[5]);
};

#endif // sampler_h__
