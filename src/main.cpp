#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

#include <astro/util.h>
#include <astro/math.h>
#include <astro/useall.h>

static const int NBANDS = 5;

struct TSED
{
	float Mr, FeH;
	float v[NBANDS];	// Mu, Mg, Mr, Mi, Mz
};
typedef std::map<float, std::map<float, TSED> > TSEDs;	// map from (Mr, FeH) -> SED

void load_seds(TSEDs &seds, const std::string &fn)
{
	std::ifstream in(fn.c_str());
	if(!in) { abort(); }

	std::string line;
	int nSEDs = 0;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment

		TSED sed;
		std::istringstream ss(line);
		ss >> sed.Mr >> sed.FeH >> sed.v[0] >> sed.v[1] >> sed.v[3] >> sed.v[4];

		// playing with thinning the SED library
		//{float k = sed.Mr * 20; if(k - int(k) != 0) { continue; } }
		//{float k = sed.FeH * 20; if(k - int(k) != 0) { continue; } }

		sed.v[2] = sed.Mr;			// Mr
		sed.v[1] += sed.v[2];			// Mg
		sed.v[0] += sed.v[1];			// Mu
		sed.v[3] = sed.v[2] - sed.v[3];		// Mi
		sed.v[4] = sed.v[3] - sed.v[4];		// Mz

		seds[sed.Mr][sed.FeH] = sed;
		nSEDs++;
	}

	std::cerr << "Loaded SEDs: " << nSEDs << " SEDs total, for " << seds.size() << " distinct absolute magnitudes.\n";
}

struct TLF	// the luminosity function
{
	float Mr0, dMr;
	std::vector<float> lf;

	float operator()(float Mr) const	// return the LF at position Mr (nearest neighbor interp.)
	{
		int idx = (int)floor((Mr - Mr0) / dMr + 0.5);
		if(idx < 0) { return lf.front(); }
		if(idx >= lf.size()) { return lf.back(); }
		return lf[idx];
	}

	void load(const std::string &fn)
	{
		std::ifstream in(fn.c_str());
		if(!in) { abort(); }

		dMr = -1;
		lf.clear();

		std::string line;
		float Mr, Phi;
		while(std::getline(in, line))
		{
			if(!line.size()) { continue; }		// empty line
			if(line[0] == '#') { continue; }	// comment

			std::istringstream ss(line);
			ss >> Mr >> Phi;

				if(dMr == -1) { Mr0 = Mr; dMr = 0; }
			else	if(dMr == 0)  { dMr = Mr - Mr0; }

			lf.push_back(Phi);
		}

		std::cerr << "Loaded LF: " << Mr0 << " <= Mr <= " <<  Mr0 + dMr*(lf.size()-1) << "\n";
	}
};

struct TModel
{
	// Model parameters: galactic ctr. distance, solar offset, disk scale height & length
	float R0, Z0, H0, L0;

	TModel(float R0_, float Z0_, float L0_, float H0_) : R0(R0_), Z0(Z0_), L0(L0_), H0(H0_) {}

	// XYZ are assumed to be Galactocentric
	inline float expdisk(float X, float Y, float Z)
	{
		float R = sqrt(X*X + Y*Y);
		float rho = exp(-(fabs(Z + Z0) - fabs(Z0))/H0 + -(R-R0)/L0);

		return rho;
	}

	// returns Galactocentric XYZ given l,b,d
	inline void computeCartesianPositions(float &X, float &Y, float &Z, double ldeg, double bdeg, double d)
	{
		Radians l = rad(ldeg), b = rad(bdeg);

		X = R0 - cos(l)*cos(b)*d;
		Y = -sin(l)*cos(b)*d;
		Z = sin(b)*d;
	}

	// the number of stars per unit solid area and unit distance modulus,
	// in direction l,b at distance modulus DM
	inline float dn(double l, double b, float DM)
	{
		float X, Y, Z;
		float D = pow10(DM/5.+1.);
		computeCartesianPositions(X, Y, Z, l, b, D);

		float rho = expdisk(X, Y, Z);
		float dn = rho * log(10.)/5 * pow(D, 3);

		return dn;
	}
};

// Computing the marginalized log-likelihood (up to an additive constant!)
// of the SED sed given the observed SED M[] and its errors sigma[]
double log_likelihood(float M[NBANDS], float sigma[NBANDS], const TSED &sed)
{
	// likelihoods are independent gaussians
	static const float sqrt2 = sqrt(2);
	//static const float logSqrtTwoPi = -0.5f * log(ctn::twopi);

	float logLtotal = 0;
	FOR(0, NBANDS)
	{
		float x = (M[i] - sed.v[i]) / (sqrt2*sigma[i]);
		//if(fabs(x) > 10) { x = 10; } // widen the tails of the Gaussian (to allow for outliers)

		#if 0
		float logN = logSqrtTwoPi - log(sigma[i]);
		float logL = logN - x*x;
		#else
		float logL = -x*x;
		#endif
		logLtotal += logL;

		// optimization: stop immediately if very unlikely
		if(logLtotal < -100) { return logLtotal; }
	}

	return logLtotal;
}

/*
4.19  -2.50  0.6739  0.2143  0.0633  -0.0241  -0.0339
             ug      gr      ri      iz       zy

M=          5.0782  4.4043  4.19  4.2141 4.248
DM=15

20.0782 19.4043 19.1900 19.2141 19.2480
*/

int main(int argc, char **argv)
{
	TSEDs seds;
	load_seds(seds, "MSandRGBcolors_v1.3.dat");

	TLF lf;
	lf.load("MrLF.MSandRGB_v1.0.dat");

	TModel model(8000, 25, 2150, 245);

	// read observation from stdin
	float m[NBANDS], sigma[NBANDS];
//	std::istringstream ss("20.0782 19.4043 19.1900 19.2141 19.2480  0.1 0.03 0.03 0.03 0.03");
	std::istringstream ss("17.4115 15.6745 15.0000 14.7138 14.5678  0.1 0.03 0.03 0.03 0.03");
	FOR(0, NBANDS) { ss >> m[i]; }
	FOR(0, NBANDS) { ss >> sigma[i]; }
	FOR(0, NBANDS)
	{
		std::cerr << m[i] << " +/- " << sigma[i] << "; ";
	}
	std::cerr << "\n";

	double l = 0, b = 90;
//	for(double l = 0; l < 360; l += 1)
	{
		double Ltot = 0;
		for(float DM=5; DM <= 20; DM += 0.1)
		{
			double pDM__G = model.dn(l, b, DM);

			float M[NBANDS];
			FOR(0, NBANDS) { M[i] = m[i] - DM; }

			FOREACH(seds)
			{
				float Mr = i->first;
				double pMr__DM_G = lf(Mr);

				float logPrior = log(pDM__G) + log(pMr__DM_G);
				FOREACHj(j, i->second)
				{
					TSED &sed = j->second;
					float logL = log_likelihood(M, sigma, sed);
					if(logL <= -100) { continue; }

					float logP = logL + logPrior;
					#if 1
						std::cout << DM << "\t" << sed.Mr << "\t" << sed.FeH << "\t" << logL << "\t" << logP << "\n";
					#endif
					Ltot += expf(logL);
				}
			}

	//		std::cerr << "DM=" << DM << "\n";
		}
		std::cerr << "Total likelihood: " << Ltot << "\n";
	}

	return 0;
}
