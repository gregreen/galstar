#include "sampler.h"

#include <iostream>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

#include <boost/format.hpp>

#include <astro/util.h>
#include <astro/math.h>
#include <astro/useall.h>

//////////////////////////////////////////////////////////////////////////////////////
// I/O
//////////////////////////////////////////////////////////////////////////////////////

void load_seds(TSEDs &seds, const std::string &fn)
{
	std::ifstream in(fn.c_str());
	if(!in) { std::cerr << "Could not read SEDs from '" << fn << "'\n"; abort(); }

	std::string line;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		TSED sed;
		std::istringstream ss(line);
		ss >> sed.Mr >> sed.FeH >> sed.v[0] >> sed.v[1] >> sed.v[3] >> sed.v[4];

		// playing with thinning the SED library
		//{double k = sed.Mr * 20; if(k - int(k) != 0) { continue; } }
		//{double k = sed.FeH * 20; if(k - int(k) != 0) { continue; } }

		sed.v[2]  = sed.Mr;			// Mr
		sed.v[1] += sed.v[2];			// Mg
		sed.v[0] += sed.v[1];			// Mu
		sed.v[3]  = sed.v[2] - sed.v[3];	// Mi
		sed.v[4]  = sed.v[3] - sed.v[4];	// Mz

		seds.push_back(sed);
	}
	std::sort(seds.begin(), seds.end());

	std::cerr << "# Loaded " << seds.size() << " SEDs from " << fn << "\n";
}

// get an SED producing a closest match to (Mr,FeH)
TSEDs::iterator get_closest_SED(TSEDs &seds, double Mr, double FeH)
{
	// WARNING: This is a really, really, really, slow search algorithm
	TSEDs::iterator best;
	double dmin = 1e10;
	FOREACH(seds)
	{
		double d = sqr(i->Mr - Mr) + sqr(i->FeH - FeH);
		if(d < dmin) { dmin = d; best = i; }
	}
	return best;
}

void TLF::load(const std::string &fn)
{
	std::ifstream in(fn.c_str());
	if(!in) { std::cerr << "Could not read LF from '" << fn << "'\n"; abort(); }

	dMr = -1;
	lf.clear();

	std::string line;
	double Mr, Phi;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		std::istringstream ss(line);
		ss >> Mr >> Phi;

			if(dMr == -1) { Mr0 = Mr; dMr = 0; }
		else	if(dMr == 0)  { dMr = Mr - Mr0; }

		lf.push_back(log(Phi));
	}

	std::cerr << "# Loaded Phi(" << Mr0 << " <= Mr <= " <<  Mr0 + dMr*(lf.size()-1) << ") LF from " << fn << "\n";
}

//////////////////////////////////////////////////////////////////////////////////////
// Model
//////////////////////////////////////////////////////////////////////////////////////

TModel::TModel(const std::string &lf_fn, const std::string &seds_fn)
	:
	lf(lf_fn),
	DM_range(5., 20., .02), Ar_range(0.),
	Mr_range(ALL), FeH_range(ALL)
{
	load_seds(seds, seds_fn);
}

TModel::Params::Getter TModel::Params::varname2getter(const std::string &var)
{
	if(var == "Ar")  { return &TModel::Params::get_Ar; }
	if(var == "DM")  { return &TModel::Params::get_DM; }
	if(var == "Mr")  { return &TModel::Params::get_Mr; }
	if(var == "FeH") { return &TModel::Params::get_FeH; }
	return NULL;
}

const double TModel::Acoef[NBANDS] = {5.155/2.751, 3.793/2.751, 1., 2.086/2.751, 1.479/2.751};

// compute Galactocentric XYZ given l,b,d
void TModel::computeCartesianPositions(double &X, double &Y, double &Z, double ldeg, double bdeg, double d) const
{
	peyton::Radians l = rad(ldeg), b = rad(bdeg);

	X = R0 - cos(l)*cos(b)*d;
	Y = -sin(l)*cos(b)*d;
	Z = sin(b)*d;
}

double TModel::expdisk(double X, double Y, double Z) const
{
	double R = sqrt(X*X + Y*Y);
	double rho = exp(-(fabs(Z + Z0) - fabs(Z0))/H1 + -(R-R0)/L1);

	return rho;
}

// the number of stars per unit solid area and unit distance modulus,
// in direction l,b at distance modulus DM
double TModel::dn(double l, double b, double DM) const
{
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, l, b, D);

	double rho  = expdisk(X, Y, Z);
	double dn = rho * pow(D, 3.);

	return dn;
}

//////////////////////////////////////////////////////////////////////////////////////
// Grid sampler
//////////////////////////////////////////////////////////////////////////////////////

// Computing the log-likelihood (up to an additive constant!)
// of the SED sed given the observed SED M[] and its errors sigma[]
inline double logL_SED(const double M[NBANDS], const double sigma[NBANDS], const TSED &sed)
{
	// likelihoods are independent gaussians
	double logLtotal = 0;
	FOR(0, NBANDS)
	{
		double x = (M[i] - sed.v[i]) / sigma[i];
		logLtotal -= x*x;

		// optimization: stop immediately if SED is very unlikely
		if(logLtotal < -200) { break; }
	}

	return 0.5*logLtotal;
}

// evenly sample the (DM, Ar, SEDs) space, and pass the posterior
// probability to the Marginalizer object
void TModel::sample(Marginalizer &out, double l, double b, const double m[NBANDS], const double err[NBANDS])
{
	Params p;
	FORRANGEj(DM, DM_range)
	{
		std::cerr << ".";
		p.DM = *DM;
		double logP_DM = log( dn(l, b, p.DM) );	// probability that the star is at DM

		FORRANGEj(Ar, Ar_range)
		{
			p.Ar = *Ar;
			double logP_Ar = log(1.);		// probability that extinction is Ar (flat prior, for now)

			// Compute the absolute magnitudes of this object,
			// given Ar, DM and m
			double M[NBANDS];
			FOR(0, NBANDS) { M[i] = m[i] - p.DM - p.Ar*Acoef[i]; }

			double cache_Mr = seds.begin()->Mr;
			double logP_SED = lf(cache_Mr);
			FOREACH(seds)
			{
				p.SED = &*i;

				// skip seds out of given range
				if(!Mr_range.contains(p.SED->Mr) || !FeH_range.contains(p.SED->FeH))
				{
					continue;
				}

				// simple caching, relies on the above FOREACH looping
				// over FeH first, Mr second
				if(p.SED->Mr != cache_Mr)
				{
					cache_Mr = p.SED->Mr;
					logP_SED = lf(cache_Mr);
				}

				double logL = logL_SED(M, err, *p.SED);
				if(logL <= -100) { continue; }	// optimization

				double logP = logL + logP_SED + logP_DM + logP_Ar;
				out(p, exp(logP));
			}
		}
	}
	std::cerr << "\n";
}
