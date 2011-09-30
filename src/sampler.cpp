#include "sampler.h"
#include <omp.h>

#include <iostream>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <iomanip>
#include <limits>

#include <boost/format.hpp>

#include <astro/util.h>
#include <astro/math.h>
#include <astro/useall.h>

//////////////////////////////////////////////////////////////////////////////////////
// I/O
//////////////////////////////////////////////////////////////////////////////////////

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
	unsigned int Mr_index, FeH_index;
	dFeH = 0.05;
	dMr = 0.01;
	FeH_min = -2.50;
	Mr_min = -1.00;
	N_FeH = 51;
	N_Mr = 2901;
	seds = new TSED[N_FeH*N_Mr];
	
	// Load the SEDs
	std::ifstream in(seds_fn.c_str());
	if(!in) { std::cerr << "Could not read SEDs from '" << seds_fn << "'\n"; abort(); }
	
	double Mr, FeH;
	std::string line;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment
		
		std::istringstream ss(line);
		ss >> Mr >> FeH;
		
		TSED* sed = get_sed(Mr, FeH);
		
		sed->Mr = Mr;
		sed->FeH = FeH;
		ss >> sed->v[0] >> sed->v[1] >> sed->v[3] >> sed->v[4];
		
		sed->v[2]  = sed->Mr;			// Mr
		sed->v[1] += sed->v[2];			// Mg
		sed->v[0] += sed->v[1];			// Mu
		sed->v[3]  = sed->v[2] - sed->v[3];	// Mi
		sed->v[4]  = sed->v[3] - sed->v[4];	// Mz
	}
	
	std::cerr << "# Loaded " << N_FeH*N_Mr << " SEDs from " << seds_fn << "\n";
}

const double TModel::Acoef[NBANDS] = {5.155/2.751, 3.793/2.751, 1., 2.086/2.751, 1.479/2.751};

// compute Galactocentric XYZ given l,b,d
inline void TModel::computeCartesianPositions(double &X, double &Y, double &Z, double cos_l, double sin_l, double cos_b, double sin_b, double d) const
{
	X = R0 - cos_l*cos_b*d;
	Y = -sin_l*cos_b*d;
	Z = sin_b*d;
}

inline double TModel::rho_halo(double R, double Z) const {
	double r_eff2 = R*R + sqr(Z/qh);
	if(r_eff2 <= R_br2) {
		return fh*pow(r_eff2/(R0*R0), nh/2.);
	} else {
		return fh_outer*pow(r_eff2/(R0*R0), nh_outer/2.);
	}
}

inline double TModel::rho_disk(double R, double Z) const {
	double rho_thin = exp(-(fabs(Z+Z0) - fabs(Z0))/H1 - (R-R0)/L1);
	double rho_thick = f*exp(-(fabs(Z+Z0) - fabs(Z0))/H2 - (R-R0)/L2);
	return rho_thin + rho_thick;
}

// the number of stars per unit solid area and unit distance modulus,
// in direction l,b at distance modulus DM
inline double TModel::log_dn(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const {
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, cos_l, sin_l, cos_b, sin_b, D);
	
	// Thin and thick disks
	double R = sqrt(X*X + Y*Y);
	double log_rho = log(rho_disk(R,Z) + rho_halo(R,Z));
	
	return log_rho + (3.*2.30258509/5.)*DM;
}

// Fraction of stars at given position in the halo
inline double TModel::f_halo(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const {
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, cos_l, sin_l, cos_b, sin_b, D);
	double R = sqrt(X*X + Y*Y);
	
	double rho_halo_tmp = rho_halo(R,Z);
	double f_h_tmp = rho_halo_tmp/(rho_disk(R,Z)+rho_halo_tmp);
	
	return f_h_tmp;
}

// Mean disk metallicity at given position in space
inline double TModel::mu_disk(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const {
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, cos_l, sin_l, cos_b, sin_b, D);
	
	// TODO: Move some of these parameters to the TModel object and allow them to be set from the commandline
	double mu_inf = -0.78;
	double delta_mu = 0.35;
	double H_mu = 1000.;
	return mu_inf + delta_mu*exp(-fabs(Z+Z0)/H_mu);
}


TSED* TModel::get_sed(const double Mr, const double FeH) const {
	unsigned int Mr_index = (unsigned int)((Mr-Mr_min)/dMr + 0.5);
	unsigned int FeH_index = (unsigned int)((FeH-FeH_min)/dFeH + 0.5);
	if((Mr_index < 0) || (Mr_index >= N_Mr) || (FeH_index < 0) || (FeH_index >= N_FeH)) { return NULL; }
	return &seds[N_Mr*FeH_index+Mr_index];
}

inline void TModel::get_sed_inline(TSED** sed_out, const double Mr, const double FeH) const {
	unsigned int Mr_index = (unsigned int)((Mr-Mr_min)/dMr + 0.5);
	unsigned int FeH_index = (unsigned int)((FeH-FeH_min)/dFeH + 0.5);
	if((Mr_index > 0) && (Mr_index < N_Mr) && (FeH_index > 0) && (FeH_index < N_FeH)) { *sed_out = &seds[N_Mr*FeH_index+Mr_index]; }
}

inline unsigned int TModel::sed_index(const double Mr, const double FeH) const {
	unsigned int Mr_index = (unsigned int)((Mr-Mr_min)/dMr + 0.5);
	unsigned int FeH_index = (unsigned int)((FeH-FeH_min)/dFeH + 0.5);
	return N_Mr*FeH_index+Mr_index;
}

// Metallicity distribution of stars at elevation <Z> above the Galactic plane
inline double TModel::log_p_FeH(double cos_l, double sin_l, double cos_b, double sin_b, double DM, double FeH) const {
	#define sqrttwopi 2.50662827
	double f_H = f_halo(cos_l, sin_l, cos_b, sin_b, DM);
	
	// Halo
	double mu_H = -1.46;
	double sigma_H = 0.3;
	double P_tmp = f_H * exp(-sqr(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (sqrttwopi*sigma_H);
	
	// Metal-poor disk
	double mu_D = mu_disk(cos_l, sin_l, cos_b, sin_b, DM) - 0.067;
	double sigma_D = 0.2;
	P_tmp += 0.67 * (1-f_H) * exp(-sqr(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (sqrttwopi*sigma_D);
	
	// Metal-rich disk
	double mu_D_poor = mu_D + 0.14;
	double sigma_D_poor = 0.2;
	P_tmp += 0.37 * (1-f_H) * exp(-sqr(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (sqrttwopi*sigma_D_poor);
	#undef sqrt2pi
	
	return log(P_tmp);
}

// Metallicity distribution of stars at distance modulus <DM> along line of sight defined in MCMCParams object
inline double MCMCParams::log_p_FeH_fast(double DM, double FeH) {
	#define sqrttwopi 2.50662827
	double f_H = f_halo_interp(DM);
	
	// Halo
	double mu_H = -1.46;
	double sigma_H = 0.3;
	double P_tmp = f_H * exp(-(FeH-mu_H)*(FeH-mu_H)/(2.*sigma_H*sigma_H)) / (sqrttwopi*sigma_H);
	
	// Metal-poor disk
	double mu_D = mu_disk_interp(DM) - 0.067;
	double sigma_D = 0.2;
	P_tmp += 0.63 * (1-f_H) * exp(-(FeH-mu_D)*(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (sqrttwopi*sigma_D);
	
	// Metal-rich disk
	double mu_D_poor = mu_D + 0.14;
	double sigma_D_poor = 0.2;
	P_tmp += 0.37 * (1-f_H) * exp(-(FeH-mu_D_poor)*(FeH-mu_D_poor)/(2.*sigma_D_poor*sigma_D_poor)) / (sqrttwopi*sigma_D_poor);
	#undef sqrt2pi
	
	return log(P_tmp);
}

// Computing the log-likelihood (up to an additive constant!)
// of the SED sed given the observed SED M[] and its errors sigma[]
inline double logL_SED(const double (&M)[NBANDS], const double (&sigma)[NBANDS], const TSED &sed)
{
	// likelihoods are independent gaussians
	double logLtotal = 0;
	double x;
	for(unsigned int i=0; i<NBANDS; i++) {
		x = (M[i] - sed.v[i]) / sigma[i];
		logLtotal -= x*x;
	}

	return 0.5*logLtotal;
}

//////////////////////////////////////////////////////////////////////////////////////
// MCMC sampler
//////////////////////////////////////////////////////////////////////////////////////

// pdf for the MCMC routine
inline double calc_logP(const double *const x, size_t dim, MCMCParams &p) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	double Ar, DM;
	
	// P(Ar|G): Flat prior for Ar>0
	if(x[_Ar] < 0.) { logP -= 10*x[_Ar]*x[_Ar]; Ar = 0.; } else { Ar = x[_Ar]; }
	
	// P(Mr|G) from luminosity function
	logP += p.model.lf(x[_Mr]);
	
	// P(DM|G) from model of galaxy
	if(x[_DM] < 0.) { logP -= 10*x[_DM]*x[_DM]; DM = 0.; } else { DM = x[_DM]; }
	
	logP += p.log_dn_interp(DM);
	
	// P(FeH|DM,G) from Ivezich et al (2008)
	logP += p.log_p_FeH_fast(DM, x[_FeH]);
	
	
	// P(g,r,i,z,y|Ar,Mr,DM) from model spectra
	double M[NBANDS];
	FOR(0, NBANDS) { M[i] = p.m[i] - DM - x[_Ar]*p.model.Acoef[i]; }	// Calculate absolute magnitudes from observed magnitudes, distance and extinction
	TSED* closest_sed = NULL;
	p.model.get_sed_inline(&closest_sed, x[_Mr], x[_FeH]);			// Retrieve template spectrum
	if(closest_sed == NULL) { logP -= 1000; } else { logP += logL_SED(M, p.err, *closest_sed); }
	
	#undef neginf
	return logP;
}

// Generates a random state, with a flat distribution in each parameter
void ran_state(double *x_0, size_t dim, gsl_rng *r, MCMCParams &p) {
	x_0[_DM] = gsl_ran_flat(r, 6, 15);
	x_0[_Ar] = gsl_ran_flat(r, 0.1, 2.9);
	x_0[_Mr] = gsl_ran_flat(r, -0.9, 27.9);
	x_0[_FeH] = gsl_ran_flat(r, -2.4, -0.1);
}

bool sample_mcmc(TModel &model, double l, double b, typename TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats)
{
	unsigned int N_threads = 4;		// # of parallel chains to run
	unsigned int N_burn_in = 1000;		// # of burn-in steps
	unsigned int N_steps = 10000;		// # of steps to take per round
	unsigned int L = 100;			// # of timesteps per integration
	double eta = 1e-5;			// Integration step size
	double target_acceptance = 0.98;	// Target acceptance rate for Monte Carlo steps
	unsigned int max_rounds = 20;		// After <max_rounds> rounds, the Markov chains are terminated
	double convergence_threshold = 1.05;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.1;	// Return false if GR diagnostic is above this level at end of run
	bool convergence = true;
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set run parameters
	MCMCParams p(l, b, mag, model);
	typename THybridMC<MCMCParams, TMultiBinner<4> >::log_pdf_t pdf_ptr = &calc_logP;
	typename THybridMC<MCMCParams, TMultiBinner<4> >::rand_state_t rand_state_ptr = &ran_state;
	TParallelHybridMC<MCMCParams, TMultiBinner<4> > sampler(N_threads, 4, pdf_ptr, rand_state_ptr, p, multibinner, stats);
	
	// Burn in and tune integration step size
	sampler.tune(L, eta, target_acceptance, 50);
	std::cerr << "# eta -> " << eta << std::endl;
	sampler.step_multiple(N_burn_in, L, eta, false);
	std::cerr << "# Burn-in acceptance rate: " << sampler.acceptance_rate() << " (target = " << target_acceptance << ")" << std::endl;
	sampler.tune(L, eta, target_acceptance, 50);
	std::cerr << "# eta -> " << eta << std::endl;
	
	// Main run
	double bail = false;
	unsigned int count = 0;
	while(!bail) {
		sampler.step_multiple(N_steps, L, eta);
		bail = true;
		sampler.calc_GR_stat();
		for(unsigned int i=0; i<4; i++) {
			if(sampler.get_GR_stat(i) > convergence_threshold) { bail = false; break; }
		}
		count++;
		if(count >= max_rounds) { bail = true; }
	}
	//stats = sampler.get_stats();
	
	// Flag conconvergence
	for(unsigned int i=0; i<4; i++) {
		if(sampler.get_GR_stat(i) > nonconvergence_flag) { convergence = false; break; }
	}
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) { multibinner.get_binner(i)->normalize(); }	// Normalize bins to peak value
	
	// Print stats and run time
	sampler.print_stats();
	if(!convergence) { std::cout << std::endl << "Did not converge." << std::endl; }
	std::cout << std::endl << "Time elapsed for " << N_steps*count << " steps (" << count << " rounds) on " << N_threads << " threads: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return convergence;
}
