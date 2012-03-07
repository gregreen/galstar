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
	
	lf_interp = new TLinearInterp(Mr0, Mr0 + dMr*(lf.size()-1), lf.size());
	for(unsigned int i=0; i<lf.size(); i++) { (*lf_interp)[i] = lf[i]; }
	
	std::cerr << "# Loaded Phi(" << Mr0 << " <= Mr <= " <<  Mr0 + dMr*(lf.size()-1) << ") LF from " << fn << "\n";
}

//////////////////////////////////////////////////////////////////////////////////////
// Model
//////////////////////////////////////////////////////////////////////////////////////

TModel::TModel(const std::string &lf_fn, const std::string &seds_fn)
	: lf(lf_fn), DM_range(5., 20., .02), Ar_range(0.), Mr_range(ALL), FeH_range(ALL), sed_interp(NULL)
{
	unsigned int Mr_index, FeH_index;
	dFeH = 0.05;
	dMr = 0.01;
	FeH_min = -2.50;
	Mr_min = -1.00;
	N_FeH = 51;
	N_Mr = 2901;
	FeH_max = FeH_min + dFeH*(N_FeH-1);
	Mr_max = Mr_min + dMr*(N_Mr-1);
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
	
	sed_interp = new TBilinearInterp<TSED>(Mr_min, Mr_max, N_Mr, FeH_min, FeH_max, N_FeH);
	unsigned int idx;
	for(unsigned int i=0; i<N_Mr*N_FeH; i++) {
		idx = sed_interp->get_index(seds[i].Mr, seds[i].FeH);
		(*sed_interp)[idx] = seds[i];
	}
	
	std::cerr << "# Loaded " << N_FeH*N_Mr << " SEDs from " << seds_fn << "\n";
}

const double TModel::Acoef[NBANDS] = {1.8236, 1.4241, 1.0000, 0.7409, 0.5821}; //{5.155/2.751, 3.793/2.751, 1., 2.086/2.751, 1.479/2.751};

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
double TModel::log_dn(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const {
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, cos_l, sin_l, cos_b, sin_b, D);
	
	// Thin and thick disks
	double R = sqrt(X*X + Y*Y);
	double log_rho = log(rho_disk(R,Z) + rho_halo(R,Z));
	
	return log_rho + (3.*2.30258509/5.)*DM;
}

// Fraction of stars at given position in the halo
double TModel::f_halo(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const {
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, cos_l, sin_l, cos_b, sin_b, D);
	double R = sqrt(X*X + Y*Y);
	
	double rho_halo_tmp = rho_halo(R,Z);
	double rho_disk_tmp = rho_disk(R,Z);
	double f_h_tmp = rho_halo_tmp/(rho_disk_tmp+rho_halo_tmp);
	
	return f_h_tmp;
}

// Mean disk metallicity at given position in space
double TModel::mu_disk(double cos_l, double sin_l, double cos_b, double sin_b, double DM) const {
	double X, Y, Z;
	double D = pow10(DM/5.+1.);
	computeCartesianPositions(X, Y, Z, cos_l, sin_l, cos_b, sin_b, D);
	
	// TODO: Move some of these parameters to the TModel object and allow them to be set from the commandline
	double mu_inf = -0.82;
	double delta_mu = 0.55;
	double H_mu = 500.;
	return mu_inf + delta_mu*exp(-fabs(Z+Z0)/H_mu);
}


TSED* TModel::get_sed(double Mr, double FeH) const {
	unsigned int Mr_index = (unsigned int)((Mr-Mr_min)/dMr + 0.5);
	unsigned int FeH_index = (unsigned int)((FeH-FeH_min)/dFeH + 0.5);
	if((Mr_index < 0) || (Mr_index >= N_Mr) || (FeH_index < 0) || (FeH_index >= N_FeH)) { return NULL; }
	return &seds[N_Mr*FeH_index+Mr_index];
}

inline void TModel::get_sed_inline(TSED** sed_out, const double Mr, const double FeH) const {
	unsigned int Mr_index = (unsigned int)((Mr-Mr_min)/dMr + 0.5);
	unsigned int FeH_index = (unsigned int)((FeH-FeH_min)/dFeH + 0.5);
	if((Mr_index > 0) && (Mr_index < N_Mr) && (FeH_index > 0) && (FeH_index < N_FeH)) {
		*sed_out = &seds[N_Mr*FeH_index+Mr_index];
	} else {
		*sed_out = NULL;
	}
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
	P_tmp += 0.63 * (1-f_H) * exp(-sqr(FeH-mu_D)/(2.*sigma_D*sigma_D)) / (sqrttwopi*sigma_D);
	
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
	#undef sqrttwopi
	
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


// Generates a random state for an entire line of sight, with a flat distribution in each parameter
void ran_state_los(double *const x_0, unsigned int N, gsl_rng *r, MCMCParams &p) {
	unsigned int N_stars = N/4;
	double dDM_max = 18. / (double)N_stars;
	double dAr_max = 5. / (double)N_stars;
	for(unsigned int i=0; i<N_stars; i++) {
		x_0[4*i+_DM] = gsl_ran_flat(r, 0.05, dDM_max);
		x_0[4*i+_Ar] = gsl_ran_flat(r, 0.05, dAr_max);
		x_0[4*i+_Mr] = gsl_ran_flat(r, -0.9, 27.9);
		x_0[4*i+_FeH] = gsl_ran_flat(r, -2.4, -0.1);
	}
	x_0[_DM] += 5.1;	// Lowest reasonable DM
}

// Unnormalized posterior for entire line of sight:
// P(DM,Ar,FeH,Mr|g,r,i,z,y,G)
double calc_logP_los(const double *const x, unsigned int N, MCMCParams &p) {
	unsigned int N_stars = N/4;
	gsl_permutation *data_order = gsl_permutation_calloc(N_stars);
	
	double logP = log_prior(x, N, p);
	if(logP == -std::numeric_limits<double>::infinity()) { return logP; }
	
	unsigned int N_permutations = gsl_sf_fact(N_stars);
	double low_cutoff = -std::numeric_limits<double>::infinity() / N_stars;
	double logL_i;
	for(unsigned int i=0; i<N_permutations; i++) {
		logL_i = log_permutation_likelihood(x, N, p, data_order);
		if(logL_i < low_cutoff) { logP += low_cutoff; } else { logP += logL_i; }
		gsl_permutation_next(data_order);
	}
	
	gsl_permutation_free(data_order);
	
	return logP;
}

// P(DM,Ar,FeH,Mr|G)
double log_prior(const double *const x, unsigned int N, MCMCParams &p) {
	unsigned int N_stars = N/4;
	
	// Check that DM, Ar are in increasing order
	for(unsigned int i=0; i<N_stars; i++) {
		// P(Ar|G) = 0 for Ar < 0, stars are ordered by increasing reddening
		if(x[4*i+_Ar] < 0.) { return -std::numeric_limits<double>::infinity(); }
		// P(DM|G) = 0 for DM < 0, stars are ordered by increasing distance
		if(x[4*i+_DM] < 0.) { return -std::numeric_limits<double>::infinity(); }
		// Mr and FeH must be within the range contained in the SED template library
		if((x[_Mr] < p.model.Mr_min) || (x[_Mr] > p.model.Mr_max) || (x[_FeH] < p.model.FeH_min) || (x[_FeH] > p.model.FeH_max)) { return -std::numeric_limits<double>::infinity(); }
	}
	
	// Sum priors for each star
	double DM = 0.;
	double Ar = 0.;
	double logP_0;
	double logP_i;
	double tmp_sum = 0.;
	for(unsigned int i=0; i<N_stars; i++) {
		DM += x[4*i+_DM];
		Ar += x[4*i+_Ar];
		
		logP_i = 0;
		// P(FeH|DM,G) from Ivezich et al (2008)
		logP_i += p.log_p_FeH_fast(DM, x[4*i+_FeH]);
		// P(DM|G) from model of galaxy
		logP_i += p.log_dn_interp(DM);
		// P(Mr|G) from luminosity function
		logP_i += p.model.lf(x[4*i+_Mr]);
		
		if(i == 0) { logP_0 = logP_i; } else { tmp_sum += exp(logP_i-logP_0); }
	}
	
	return logP_0 + log(tmp_sum);
	
	return 0.;
}

// P(g,r,i,z,y|Ar,Mr,DM) from model spectra for a given permutation
double log_permutation_likelihood(const double *const x, unsigned int N, MCMCParams &p, gsl_permutation *data_order) {
	double logL = 0.;
	
	double M[NBANDS];		// Absolute magnitudes
	//TSED* closest_sed = NULL;	// Closest match in SED template library
	TSED sed_bilin_interp;
	
	unsigned int N_stars = N/4;
	TStellarData::TMagnitudes *mag = NULL;
	
	double DM = 0.;
	double Ar = 0.;
	for(unsigned int i=0; i<N_stars; i++) {
		DM += x[4*i+_DM];
		Ar += x[4*i+_Ar];
		unsigned int idx = gsl_permutation_get(data_order, i);
		mag = &p.data[gsl_permutation_get(data_order, i)];					// Get ith star in the given permutation
		for(unsigned int k=0; k<NBANDS; k++) { M[k] = mag->m[k] - DM - Ar*p.model.Acoef[k]; }	// Calculate absolute magnitudes from observed magnitudes, distance and extinction
		TSED sed_bilin_interp = (*p.model.sed_interp)(x[_Mr], x[_FeH]);
		double logL_i = logL_SED(M, p.err, sed_bilin_interp);
		logL += logL_SED(M, p.err, sed_bilin_interp);						// Update log likelihood from difference between absolute magnitudes and template SED magnitudes
	}
	
	if(logL > -100.) {
		#pragma omp critical (cout)
		std::cout << x[_DM] << "\t" << x[_Ar] << "\t" << x[_Mr] << "\t" << x[_FeH] << "\t" << logL << std::endl;
	}
	
	return logL;
}

// TODO: Eliminate dependence on <mag>
// N_threads	 # of parallel Normal Kernel couplers to run
bool sample_mcmc_los(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TStellarData &data, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_steps=15000, unsigned int N_threads=4)
{
	unsigned int size = 20;			// # of chains in each Normal Kernel Coupler
	N_steps *= size;			// # of steps to take in each Normal Kernel Coupler per round
	unsigned int max_rounds = 10;		// After <max_rounds> rounds, the Markov chains are terminated
	unsigned int max_attempts = 1;		// Maximum number of initial seedings to attempt
	double convergence_threshold = 1.1;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.2;	// Return false if GR diagnostic is above this level at end of run
	bool convergence;
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	unsigned int N_stars = data.star.size();
	
	// Set the initial step size for the MCMC run
	double *scale_0 = new double[4*N_stars];
	double scale_norm = 100*sqrt((double)N_stars);
	for(unsigned int i=0; i<N_stars; i++) {
		scale_0[4*i+_DM] = 1. / scale_norm;
		scale_0[4*i+_Ar] = 0.2 / scale_norm;
		scale_0[4*i+_Mr] = 1. / scale_norm;
		scale_0[4*i+_FeH] = 0.2 / scale_norm;
	}
	
	// Set run parameters
	MCMCParams p(l, b, mag, model, data);
	TNKC<MCMCParams, TMultiBinner<4> >::pdf_t pdf_ptr = &calc_logP_los;
	TNKC<MCMCParams, TMultiBinner<4> >::rand_state_t rand_state_ptr = &ran_state_los;
	
	unsigned int count, tmp_max_rounds;
	double highest_GR, tmp_GR;
	for(unsigned int n=0; n<max_attempts; n++) {
		TParallelNKC<MCMCParams, TMultiBinner<4> > sampler(pdf_ptr, rand_state_ptr, 4*N_stars, size, scale_0, p, multibinner, N_threads);
		
		// Run Markov chains
		sampler.burn_in(100, 50*size, 0.18, false);
		sampler.set_bandwidth(0.1);
		//for(unsigned int i=0; i<N_threads; i++) { std::cout << "h[" << i << "] = " << sampler.get_chain(i)->get_bandwidth() << std::endl; }
		count = 0;
		convergence = false;
		tmp_max_rounds = max_rounds;
		while((count < tmp_max_rounds) && !convergence) {
			sampler.step(N_steps);
			highest_GR = -1.;
			for(unsigned int i=0; i<4; i++) {
				tmp_GR = sampler.get_GR_diagnostic(i);
				if(tmp_GR > highest_GR) { highest_GR = tmp_GR; }
				//if(sampler.get_GR_diagnostic(i) > convergence_threshold) { convergence = false; break; }
			}
			if(highest_GR < convergence_threshold) { convergence = true; }
			count++;
			if(!convergence && (count == max_rounds) && (highest_GR < 2.) && (tmp_max_rounds == max_rounds)) { tmp_max_rounds *= 2; }
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
	std::cout << std::endl << "Time elapsed for " << stats.get_N_items()/size << " steps (" << count << " rounds) on " << N_threads << " threads: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	delete[] scale_0;
	
	return convergence;
}























// pdf for individual star
inline double calc_logP(const double *const x, unsigned int N, MCMCParams &p) {
	#define neginf -std::numeric_limits<double>::infinity()
	double logP = 0.;
	
	//double x_tmp[4] = {x[0],x[1],x[2],x[3]};
	
	// P(Ar|G): Flat prior for Ar > 0. Don't allow DM < 0
	if((x[_Ar] < 0.) || (x[_DM] < 0.)) { return neginf; }
	
	// Make sure star is in range of template spectra
	if((x[_Mr] < p.model.Mr_min) || (x[_Mr] > p.model.Mr_max) || (x[_FeH] < p.model.FeH_min) || (x[_FeH] > p.model.FeH_max)) { return neginf; }
	
	// If the giant or dwarf flag is set, make sure star is appropriate type
	if(p.giant_flag == 1) {		// Dwarfs only
		if(x[_Mr] < 4.) { return neginf; }
	} else if(p.giant_flag == 2) {	// Giants only
		if(x[_Mr] > 4.) { return neginf; }
	}
	
	// P(Mr|G) from luminosity function
	double loglf_tmp = p.model.lf(x[_Mr]);
	logP += loglf_tmp;
	
	// P(DM|G) from model of galaxy
	double logdn_tmp = p.log_dn_interp(x[_DM]);
	logP += logdn_tmp;
	
	// P(FeH|DM,G) from Ivezich et al (2008)
	double logpFeH_tmp = p.log_p_FeH_fast(x[_DM], x[_FeH]);
	logP += logpFeH_tmp;
	
	// P(g,r,i,z,y|Ar,Mr,DM) from model spectra
	double M[NBANDS];
	FOR(0, NBANDS) { M[i] = p.m[i] - x[_DM] - x[_Ar]*p.model.Acoef[i]; }	// Calculate absolute magnitudes from observed magnitudes, distance and extinction
	
	//TSED* closest_sed = NULL;
	//p.model.get_sed_inline(&closest_sed, x[_Mr], x[_FeH]);			// Retrieve template spectrum
	//if(closest_sed == NULL) { return neginf; } else { logP += logL_SED(M, p.err, *closest_sed); }
	
	//double x_tmp[4] = {x[_DM], x[_Ar], x[_Mr], x[_FeH]};
	
	TSED sed_bilin_interp = (*p.model.sed_interp)(x[_Mr], x[_FeH]);
	double logL = logL_SED(M, p.err, sed_bilin_interp);
	logP += logL;
	
	#undef neginf
	return logP;
}

// Generates a random state, with a flat distribution in each parameter
void ran_state(double *const x_0, unsigned int N, gsl_rng *r, MCMCParams &p) {
	x_0[_DM] = gsl_ran_flat(r, 5.1, 19.9);
	x_0[_Ar] = gsl_ran_flat(r, 0.1, 3.0);
	if(p.giant_flag == 0) {
		x_0[_Mr] = gsl_ran_flat(r, -0.5, 27.5);	// Both giants and dwarfs
	} else if(p.giant_flag == 1) {
		x_0[_Mr] = gsl_ran_flat(r, 4.5, 27.5);	// Dwarfs only
	} else {
		x_0[_Mr] = gsl_ran_flat(r, -0.5, 3.5);	// Giants only
	}
	x_0[_FeH] = gsl_ran_flat(r, -2.4, -0.1);
}

// N_threads	 # of parallel Normal Kernel couplers to run
bool sample_mcmc(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_samplers=15, unsigned int N_steps=15000, unsigned int N_threads=4)
{
	unsigned int size = N_samplers;		// # of chains in each Normal Kernel Coupler
	N_steps *= size;			// # of steps to take in each Normal Kernel Coupler per round
	unsigned int max_rounds = 3;		// After <max_rounds> rounds, the Markov chains are terminated
	unsigned int max_attempts = 2;		// Maximum number of initial seedings to attempt
	double convergence_threshold = 1.1;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.2;	// Return false if GR diagnostic is above this level at end of run
	bool convergence;
	
	//timespec t_start, t_end;
	//clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set the initial step size for the MCMC run
	double scale_0[4];
	scale_0[_DM] = 0.1;
	scale_0[_Ar] = 0.1;
	scale_0[_Mr] = 0.1;
	scale_0[_FeH] = 0.1;
	
	// Set run parameters
	//MCMCParams p(l, b, mag, model, data);
	p.update(mag);
	TNKC<MCMCParams, TMultiBinner<4> >::pdf_t pdf_ptr = &calc_logP;
	TNKC<MCMCParams, TMultiBinner<4> >::rand_state_t rand_state_ptr = &ran_state;
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	unsigned int count, tmp_max_rounds;
	double highest_GR, tmp_GR;
	for(unsigned int n=0; n<max_attempts; n++) {
		multibinner.clear();
		TParallelNKC<MCMCParams, TMultiBinner<4> > sampler(pdf_ptr, rand_state_ptr, 4, size, scale_0, p, multibinner, N_threads);
		
		// Run Markov chains
		sampler.set_tune_rate(1.0003);
		sampler.set_bandwidth(0.05);
		sampler.burn_in(800, 50*size, 0.18, true, true);
		//sampler.set_bandwidth(0.01);
		//sampler.step(10000*size, false);
		//for(unsigned int i=0; i<N_threads; i++) { std::cout << "h[" << i << "] = " << sampler.get_chain(i)->get_bandwidth() << std::endl; }
		
		count = 0;
		convergence = false;
		tmp_max_rounds = max_rounds;
		while((count < tmp_max_rounds) && !convergence) {
			sampler.step(N_steps);
			highest_GR = -1.;
			for(unsigned int i=0; i<4; i++) {
				tmp_GR = sampler.get_GR_diagnostic(i);
				if(tmp_GR > highest_GR) { highest_GR = tmp_GR; }
				//if(sampler.get_GR_diagnostic(i) > convergence_threshold) { convergence = false; break; }
			}
			if(highest_GR < convergence_threshold) { convergence = true; }
			count++;
			if(!convergence && (count == tmp_max_rounds) && (highest_GR < 2.) && (tmp_max_rounds < 4*max_rounds)) { tmp_max_rounds += max_rounds; std::cout << "Extending run." << std::endl; }
		}
		
		sampler.print_stats();
		stats = sampler.get_stats();
		
		if(convergence) { break; } else { std::cout << "Attempt " << n+1 << " failed." << std::endl << std::endl; }
	}
	
	//stats = sampler.get_stats();
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	// Normalize bins to peak value
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) {
		multibinner.get_binner(i)->normalize();
	}
	
	// Print stats and run time
	//sampler.print_stats();
	if(!convergence) { std::cout << std::endl << "Did not converge." << std::endl; }
	std::cout << std::endl << "Time elapsed for " << stats.get_N_items()/size << " steps (" << count << " rounds) on " << N_threads << " threads: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return convergence;
}

// N_threads	 # of parallel affine samplers to run
bool sample_affine(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, unsigned int N_samplers=100, unsigned int N_steps=15000, unsigned int N_threads=4)
{
	unsigned int size = N_samplers;		// # of chains in each affine sampler
	unsigned int max_rounds = 3;		// After <max_rounds> rounds, the Markov chains are terminated
	unsigned int max_attempts = 2;		// Maximum number of initial seedings to attempt
	double convergence_threshold = 1.1;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.2;	// Return false if GR diagnostic is above this level at end of run
	bool convergence;
	
	//timespec t_start, t_end;
	//clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set run parameters
	//MCMCParams p(l, b, mag, model, data);
	p.update(mag);
	TAffineSampler<MCMCParams, TMultiBinner<4> >::pdf_t pdf_ptr = &calc_logP;
	TAffineSampler<MCMCParams, TMultiBinner<4> >::rand_state_t rand_state_ptr = &ran_state;
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	unsigned int count, tmp_max_rounds;
	double highest_GR, tmp_GR;
	for(unsigned int n=0; n<max_attempts; n++) {
		multibinner.clear();
		TParallelAffineSampler<MCMCParams, TMultiBinner<4> > sampler(pdf_ptr, rand_state_ptr, 4, size, p, multibinner, N_threads);
		
		// Burn-in
		sampler.set_scale(5.);
		sampler.step(N_steps, false, 4.);
		sampler.clear();
		
		// Main run
		count = 0;
		convergence = false;
		tmp_max_rounds = max_rounds;
		while((count < tmp_max_rounds) && !convergence) {
			sampler.set_scale(3.);
			sampler.step(N_steps, true, 10.);
			highest_GR = -1.;
			for(unsigned int i=0; i<4; i++) {
				tmp_GR = sampler.get_GR_diagnostic(i);
				if(tmp_GR > highest_GR) { highest_GR = tmp_GR; }
				//if(sampler.get_GR_diagnostic(i) > convergence_threshold) { convergence = false; break; }
			}
			if(highest_GR < convergence_threshold) { convergence = true; }
			count++;
			if(!convergence && (count == tmp_max_rounds) && (highest_GR < 2.) && (tmp_max_rounds < 4*max_rounds)) { tmp_max_rounds += max_rounds; std::cout << "Extending run." << std::endl; }
		}
		
		sampler.print_stats();
		stats = sampler.get_stats();
		
		if(convergence) { break; } else { std::cout << "Attempt " << n+1 << " failed." << std::endl << std::endl; }
	}
	
	//stats = sampler.get_stats();
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	// Normalize bins to peak value
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) {
		multibinner.get_binner(i)->normalize();
	}
	
	// Print stats and run time
	//sampler.print_stats();
	if(!convergence) { std::cout << std::endl << "Did not converge." << std::endl; }
	std::cout << std::endl << "Time elapsed for " << stats.get_N_items()/size << " steps (" << count << " rounds) on " << N_threads << " threads: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return convergence;
}

bool sample_brute_force(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TChainLogger &chainlogger, TStats &stats, unsigned int N_samples=150, unsigned int N_threads=4) {
	double Delta[4];
	//unsigned int N_samples_indiv[4] = {200, 200, 200, 200};
	for(unsigned int i=0; i<4; i++) {
		Delta[i] = (std_bin_max(i) - std_bin_min(i)) / (double)N_samples;
		//Delta[i] = (std_bin_max(i) - std_bin_min(i)) / (double)N_samples_indiv[i];
	}
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set up model parameters
	p.update(mag);
	//MCMCParams p(l, b, mag, model, data);
	
	omp_set_num_threads(N_threads);
	#pragma omp parallel
	{
		TStats stats_i(4);
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
							double logprob = calc_logP(&(x[0]), 4, p);
							prob = exp(logprob);
							multibinner(x, prob);
							//chainlogger(x, prob);
							stats_i(x, prob*1.e6);
							/*if(logprob > -3) {
								#pragma omp critical (cout)
								{
									std::cout << logprob;
									double tmp_DM = x[0];
									double logprob2;
									for(double dDM = 0.1; dDM <= 1.; dDM += 0.1) {
										x[0] = tmp_DM + dDM;
										logprob2 = calc_logP(&(x[0]), 4, p);
										std::cout << "\t" << std::setprecision(3) << logprob2;
									}
									std::cout << std::endl;
									x[0] = tmp_DM;
								}
							}*/
						}
					}
				}
			}
		}
		#pragma omp critical (addstats)
		stats += stats_i;
	}
	
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) {
		multibinner.get_binner(i)->normalize();		// Normalize bins to peak value
		//multibinner.get_binner(i)->print_bins();
	}
	
	// Print stats and run time
	stats.print();
	std::cout << std::endl << "Time elapsed for " << N_samples*N_samples*N_samples*N_samples << " samples: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return true;
}

void print_logpdf(TModel &model, double l, double b, TStellarData::TMagnitudes &mag, TStellarData &data, double (&m)[5], double (&err)[5], double DM, double Ar, double Mr, double FeH) {
	MCMCParams p(l, b, mag, model, data);
	double x[4] = {DM, Ar, Mr, FeH};
	double tmp = calc_logP(&(x[0]), 4, p);
	std::cout << std::setprecision(3) << "(" << DM << ", " << Ar << ", " << Mr << ", " << FeH << ") -> " << tmp << std::endl << std::endl;
}