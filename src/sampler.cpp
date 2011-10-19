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

// TODO: Add in thick disk and halo terms to log_dn
// pdf for the MCMC routine
inline double calc_logP(const double (&x)[4], MCMCParams &p) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	
	// P(Ar|G): Flat prior for Ar>0
	if(x[_Ar] < 0.) { return neginf; }
	
	// P(Mr|G) from luminosity function
	logP += p.model.lf(x[_Mr]);
	
	// P(DM|G) from model of galaxy
	if(x[_DM] < 0.) { return neginf; } else { logP += p.log_dn_interp(x[_DM]); }
	
	// P(FeH|DM,G) from Ivezich et al (2008)
	logP += p.log_p_FeH_fast(x[_DM], x[_FeH]);
	
	// P(g,r,i,z,y|Ar,Mr,DM) from model spectra
	double M[NBANDS];
	FOR(0, NBANDS) { M[i] = p.m[i] - x[_DM] - x[_Ar]*p.model.Acoef[i]; }	// Calculate absolute magnitudes from observed magnitudes, distance and extinction
	TSED* closest_sed = NULL;
	p.model.get_sed_inline(&closest_sed, x[_Mr], x[_FeH]);			// Retrieve template spectrum
	if(closest_sed == NULL) { return neginf; } else { logP += logL_SED(M, p.err, *closest_sed); }
	
	#undef neginf
	return logP;
}

// Generates a random state, with a flat distribution in each parameter
void ran_state(double (&x_0)[4], gsl_rng *r, MCMCParams &p) {
	x_0[_DM] = gsl_ran_flat(r, 5.1, 19.9);
	x_0[_Ar] = gsl_ran_flat(r, 0.1, 2.9);
	x_0[_Mr] = gsl_ran_flat(r, -0.9, 27.9);
	x_0[_FeH] = gsl_ran_flat(r, -2.4, -0.1);
}

// N_threads	 # of parallel Normal Kernel couplers to run
bool sample_mcmc(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats<4> &stats, unsigned int N_steps=15000, unsigned int N_threads=4)
{
	unsigned int size = 10;			// # of chains in each Normal Kernel Coupler
	N_steps *= size;			// # of steps to take in each Normal Kernel Coupler per round
	unsigned int max_rounds = 10;		// After <max_rounds> rounds, the Markov chains are terminated
	unsigned int max_attempts = 1;		// Maximum number of initial seedings to attempt
	double convergence_threshold = 1.1;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.2;	// Return false if GR diagnostic is above this level at end of run
	bool convergence;
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set the initial step size for the MCMC run
	double scale_0[4];
	scale_0[_DM] = 1.;
	scale_0[_Ar] = 0.2;
	scale_0[_Mr] = 1.;
	scale_0[_FeH] = 0.2;
	
	// Set run parameters
	MCMCParams p(l, b, mag, model);
	TNKC<4, MCMCParams, TMultiBinner<4> >::pdf_t pdf_ptr = &calc_logP;
	TNKC<4, MCMCParams, TMultiBinner<4> >::rand_state_t rand_state_ptr = &ran_state;
	
	unsigned int count;
	for(unsigned int n=0; n<max_attempts; n++) {
		TParallelNKC<4, MCMCParams, TMultiBinner<4> > sampler(pdf_ptr, rand_state_ptr, size, scale_0, p, multibinner, N_threads);
		
		// Run Markov chains
		sampler.burn_in(100, 50*size, 0.18, false);
		sampler.set_bandwidth(0.1);
		//for(unsigned int i=0; i<N_threads; i++) { std::cout << "h[" << i << "] = " << sampler.get_chain(i)->get_bandwidth() << std::endl; }
		count = 0;
		while((count < max_rounds) && !convergence) {
			sampler.step(N_steps);
			convergence = true;
			for(unsigned int i=0; i<4; i++) {
				if(sampler.get_GR_diagnostic(i) > convergence_threshold) { convergence = false; break; }
			}
			count++;
		}
		
		sampler.print_stats();
		stats = sampler.get_stats();
		
		if(convergence) { break; } else { std::cout << "Attempt " << n+1 << " failed." << std::endl << std::endl; }
		multibinner.clear();
	}
	
	//stats = sampler.get_stats();
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) { multibinner.get_binner(i)->normalize(); }	// Normalize bins to peak value
	
	// Print stats and run time
	//sampler.print_stats();
	if(!convergence) { std::cout << std::endl << "Did not converge." << std::endl; }
	std::cout << std::endl << "Time elapsed for " << stats.N_items_tot/size << " steps (" << count << " rounds) on " << N_threads << " threads: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return convergence;
}

bool sample_brute_force(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats<4> &stats, unsigned int N_samples = 150, unsigned int N_threads=4) {
	double Delta[4];
	for(unsigned int i=0; i<4; i++) {
		Delta[i] = (std_bin_max(i) - std_bin_min(i)) / (double)N_samples;
	}
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set up model parameters
	MCMCParams p(l, b, mag, model);
	
	omp_set_num_threads(N_threads);
	#pragma omp parallel
	{
		unsigned int thread_ID = omp_get_thread_num();
		double x[4];
		double prob;
		for(unsigned int i=0; i<N_samples; i++) {
			if(i % N_threads == thread_ID) {
				x[0] = std_bin_min(0) + ((double)i + 0.5)*Delta[0];
				for(unsigned j=0; j<N_samples; j++){
					x[1] = std_bin_min(1) + ((double)j + 0.5)*Delta[1];
					for(unsigned int k=0; k<N_samples; k++) {
						x[2] = std_bin_min(2) + ((double)k + 0.5)*Delta[2];
						for(unsigned int l=0; l<N_samples; l++) {
							x[3] = std_bin_min(3) + ((double)l + 0.5)*Delta[3];
							prob = exp(calc_logP(x, p));
							#pragma omp critical (multibinner)
							{
								multibinner(x, prob);
							}
							#pragma omp critical (stats)
							{
								stats(x, prob*1.e6);
							}
						}
					}
				}
			}
		}
	}
	
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) { multibinner.get_binner(i)->normalize(); }	// Normalize bins to peak value
	
	// Print stats and run time
	stats.print();
	std::cout << std::endl << "Time elapsed for " << N_samples*N_samples*N_samples*N_samples << " samples: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return true;
}
