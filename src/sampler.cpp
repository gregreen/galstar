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
	log_lf_norm = 0.;
	lf.clear();

	std::string line;
	double Mr, Phi;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment
		
		std::istringstream ss(line);
		ss >> Mr >> Phi;
		
		if(dMr == -1) {
			Mr0 = Mr; dMr = 0;
		} else if(dMr == 0) {
			dMr = Mr - Mr0;
		}
		
		lf.push_back(log(Phi));
		log_lf_norm += Phi;
	}
	
	double Mr1 = Mr0 + dMr*(lf.size()-1);
	lf_interp = new TLinearInterp(Mr0, Mr1, lf.size());
	for(unsigned int i=0; i<lf.size(); i++) { (*lf_interp)[i] = lf[i]; }
	
	log_lf_norm *= Mr1 / (double)(lf.size());
	log_lf_norm = log(log_lf_norm);
	
	std::cerr << "# Loaded Phi(" << Mr0 << " <= Mr <= " <<  Mr0 + dMr*(lf.size()-1) << ") LF from " << fn << "\n";
}

//////////////////////////////////////////////////////////////////////////////////////
// Model
//////////////////////////////////////////////////////////////////////////////////////

TModel::TModel(const std::string &lf_fn, const std::string &seds_fn, const double (&Acoef_)[NBANDS])
	: lf(lf_fn), sed_interp(NULL)
{
	
	// Set the reddening coefficient for each band
	for(unsigned int i=0; i<NBANDS; i++){ Acoef[i] = Acoef_[i]; }
	
	// A kluge to determine whether photometry is PS1 or SDSS
	bool PS1Photometry = false;
	if(fabs(Acoef[0] - 3.172/2.271) < 1e-5) {
		PS1Photometry = true;
		std::cerr << "# PS1 photometry being used." << std::endl;
	}
	
	// Load the SEDs
	
	double Mr, FeH, dMr_tmp, dFeH_tmp;
	double Mr_last = std::numeric_limits<double>::infinity();
	double FeH_last = std::numeric_limits<double>::infinity();
	Mr_min = std::numeric_limits<double>::infinity();
	Mr_max = -std::numeric_limits<double>::infinity();
	FeH_min = std::numeric_limits<double>::infinity();
	FeH_max = -std::numeric_limits<double>::infinity();
	dMr = std::numeric_limits<double>::infinity();
	dFeH = std::numeric_limits<double>::infinity();
	
	// Do a first pass through the file to get the grid spacing and size
	std::ifstream in(seds_fn.c_str());
	if(!in) { std::cerr << "Could not read SEDs from '" << seds_fn << "'\n"; abort(); }
	std::string line;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment
		
		std::istringstream ss(line);
		ss >> Mr >> FeH;
		
		// Keep track of values needed to get grid spacing and size
		if(Mr < Mr_min) { Mr_min = Mr; }
		if(Mr > Mr_max) { Mr_max = Mr; }
		if(FeH < FeH_min) { FeH_min = FeH; }
		if(FeH > FeH_max) { FeH_max = FeH; }
		
		dMr_tmp = fabs(Mr_last - Mr);
		dFeH_tmp = fabs(FeH_last - FeH);
		if((dMr_tmp != 0) && (dMr_tmp < dMr)) { dMr = dMr_tmp; }
		if((dFeH_tmp != 0) && (dFeH_tmp < dFeH)) { dFeH = dFeH_tmp; }
		Mr_last = Mr;
		FeH_last = FeH;
	}
	
	N_Mr = (unsigned int)(round((Mr_max - Mr_min) / dMr)) + 1;
	N_FeH = (unsigned int)(round((FeH_max - FeH_min) / dFeH)) + 1;
	//std::cerr << "# " << std::endl;
	//std::cerr << "# N_Mr: " << N_Mr << std::endl << "# dMr: " << dMr << std::endl << "# " << Mr_min << " <= Mr <= " << Mr_max << std::endl;
	//std::cerr << "# N_FeH: " << N_FeH << std::endl << "# dFeH: " << dFeH << std::endl << "# " << FeH_min << " <= FeH <= " << FeH_max << std::endl;
	//std::cerr << "# " << std::endl;
	seds = new TSED[N_FeH*N_Mr];
	
	// Now do a second pass to load the SEDs
	in.clear();
	in.seekg(0, std::ios_base::beg);
	if(!in) { std::cerr << "# Could not seek back to beginning of SED file!" << std::endl; }
	unsigned int count=0;
	while(std::getline(in, line))
	{
		if(!line.size()) { continue; }		// empty line
		if(line[0] == '#') { continue; }	// comment
		
		std::istringstream ss(line);
		ss >> Mr >> FeH;
		
		TSED* sed = get_sed(Mr, FeH);
		
		sed->Mr = Mr;
		sed->FeH = FeH;
		
		if(PS1Photometry) {
			ss >> sed->v[0] >> sed->v[2] >> sed->v[3] >> sed->v[4];
			sed->v[1] = sed->Mr;			// Begin with r
			sed->v[0] += sed->v[1];			// g = (g-r) + r
			sed->v[2] = sed->v[1] - sed->v[2];	// i = r - (r-i)
			sed->v[3] = sed->v[2] - sed->v[3];	// z = i - (i-z)
			sed->v[4] = sed->v[3] - sed->v[4];	// y = z - (z-y)
		} else {
			ss >> sed->v[0] >> sed->v[1] >> sed->v[3] >> sed->v[4];
			sed->v[2]  = sed->Mr;			// Mr
			sed->v[1] += sed->v[2];			// Mg
			sed->v[0] += sed->v[1];			// Mu
			sed->v[3]  = sed->v[2] - sed->v[3];	// Mi
			sed->v[4]  = sed->v[3] - sed->v[4];	// Mz
		}
		
		count++;
	}
	in.close();
	
	// Construct the SED interpolation grid
	sed_interp = new TBilinearInterp<TSED>(Mr_min, Mr_max, N_Mr, FeH_min, FeH_max, N_FeH);
	unsigned int idx;
	for(unsigned int i=0; i<N_Mr*N_FeH; i++) {
		idx = sed_interp->get_index(seds[i].Mr, seds[i].FeH);
		(*sed_interp)[idx] = seds[i];
	}
	
	if(count != N_FeH*N_Mr) { std::cerr << "# Incomplete SED library provided (grid is sparse, i.e. missing some values of (Mr,FeH)). This may cause problems." << std::endl; }
	std::cerr << "# Loaded " << N_FeH*N_Mr << " SEDs from " << seds_fn << "\n";
}


// compute Galactocentric XYZ given l,b,d
inline void TModel::computeCartesianPositions(double &X, double &Y, double &Z, double cos_l, double sin_l, double cos_b, double sin_b, double d) const
{
	X = R0 - cos_l*cos_b*d;
	Y = -sin_l*cos_b*d;
	Z = sin_b*d;
}

inline double TModel::rho_halo(double R, double Z) const {
	double r_eff2 = R*R + sqr(Z/qh) + R_epsilon2;
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
	#undef sqrttwopi
	
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

double giant_Mr = 26.5;

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
		if(x[_Mr] < giant_Mr) { return neginf; }
	} else if(p.giant_flag == 2) {	// Giants only
		if(x[_Mr] > giant_Mr) { return neginf; }
	}
	
	if(!(p.noprior)) {
		// P(Mr|G) from luminosity function
		double loglf_tmp = p.model.lf(x[_Mr]);
		logP += loglf_tmp;
		
		// P(DM|G) from model of galaxy
		double logdn_tmp = p.log_dn_interp(x[_DM]);
		logP += logdn_tmp;
		
		// P(FeH|DM,G) from Ivezich et al (2008)
		double logpFeH_tmp = p.log_p_FeH_fast(x[_DM], x[_FeH]);
		logP += logpFeH_tmp;
	}
	
	// P(g,r,i,z,y|Ar,Mr,DM) from model spectra
	double M[NBANDS];
	FOR(0, NBANDS) { M[i] = p.m[i] - x[_DM] - x[_Ar]*p.model.Acoef[i]; }	// Calculate absolute magnitudes from observed magnitudes, distance and extinction
	
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
		x_0[_Mr] = gsl_ran_flat(r, giant_Mr + 0.5, 27.5);	// Dwarfs only
	} else {
		x_0[_Mr] = gsl_ran_flat(r, -0.5, 15.);//giant_Mr - 0.5);	// Giants only
	}
	x_0[_FeH] = gsl_ran_flat(r, -2.4, -0.1);
}

// N_threads	 # of parallel affine samplers to run
bool sample_affine(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, double &evidence, unsigned int N_samplers=100, unsigned int N_steps=15000, double p_replacement=0.1, double p_mixture=0.1, unsigned int N_threads=4) {
	unsigned int size = N_samplers;	// # of chains in each affine sampler
	unsigned int max_rounds = 1;		// After <max_rounds> rounds, the Markov chains are terminated
	unsigned int max_attempts = 1;		// Maximum number of initial seedings to attempt
	double convergence_threshold = 1.1;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.2;	// Return false if GR diagnostic is above this level at end of run
	bool convergence;
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set run parameters
	p.update(mag);
	TAffineSampler<MCMCParams, TMultiBinner<4> >::pdf_t pdf_ptr = &calc_logP;
	TAffineSampler<MCMCParams, TMultiBinner<4> >::rand_state_t rand_state_ptr = &ran_state;
	
	unsigned int count, tmp_max_rounds;
	double highest_GR, tmp_GR;
	for(unsigned int n=0; n<max_attempts; n++) {
		multibinner.clear();
		TParallelAffineSampler<MCMCParams, TMultiBinner<4> > sampler(pdf_ptr, rand_state_ptr, 4, size, p, multibinner, N_threads);
		
		// Burn-in
		sampler.set_scale(5.);
		sampler.step(N_steps/3, false, 4., 0., 0.);
		sampler.step(N_steps/3, false, 0, p_replacement, 0.);
		if(p_mixture > 1.e-5) {
			bool no_steps = true;
			while(no_steps) {
				sampler.step(N_steps/3, true, 0, p_replacement, 0.);
				no_steps = false;
				for(unsigned int i=0; i<4; i++) {
					if(sampler.get_sampler(i)->get_chain().get_total_weight() < 1.) {
						no_steps = true;
						break;
					}
				}
			}
			std::cout << "Fitting Gaussian mixture model..." << std::endl;
			sampler.init_gaussian_mixture_target(4, 50);
			std::cout << "Clusters idenfified:" << std::endl;
			sampler.print_clusters();
			std::cout << "Weight:";
			for(unsigned int i=0; i<4; i++) { std::cout << " " << sampler.get_sampler(i)->get_chain().get_total_weight(); }
			std::cout << std::endl;
		}
		sampler.step(N_steps/3, false, 0, p_replacement, p_mixture);
		sampler.clear();
		//std::cout << "Burn-in complete." << std::endl;
		
		//std::cout << std::endl << "===================================================================" << std::endl << std::endl;
		
		// Main run
		count = 0;
		convergence = false;
		tmp_max_rounds = max_rounds;
		while((count < tmp_max_rounds) && !convergence) {
			sampler.set_scale(2.);
			sampler.step(N_steps, true, 0., p_replacement, p_mixture);
			highest_GR = -1.;
			for(unsigned int i=0; i<4; i++) {
				tmp_GR = sampler.get_GR_diagnostic(i);
				if(tmp_GR > highest_GR) { highest_GR = tmp_GR; }
				//if(sampler.get_GR_diagnostic(i) > convergence_threshold) { convergence = false; break; }
			}
			if(highest_GR < convergence_threshold) { convergence = true; }
			count++;
			if(!convergence && (count == tmp_max_rounds) && (highest_GR < 2.) && (tmp_max_rounds < 3*max_rounds)) { tmp_max_rounds += max_rounds; std::cout << "Extending run." << std::endl; }
		}
		
		sampler.print_stats();
		
		if(!convergence) { std::cout << "Attempt " << n+1 << " failed." << std::endl << std::endl; }
		
		if((n+1 == max_attempts) || convergence) {
			stats = sampler.get_stats();
			
			//for(unsigned int k=0; k<1; k++) {
			//evidence = sampler.get_chain().get_ln_Z_harmonic(false, 10., 0.05, 0.1);
			double L_norm = 0.;
			for(unsigned int i=0; i<NBANDS; i++) {
				if(mag.err[i] < 1.e9) {
					L_norm += 0.918938533 + log(mag.err[i]);
				}
			}
			//evidence -= 2.3 + L_norm;	// Estimating effect of Ar prior
			//std::cout << "ln(Z) = " << evidence << std::endl << std::endl;
			
			evidence = sampler.get_chain().get_ln_Z_harmonic(true, 1., 1., 0.01);
			evidence -= 2.3 + L_norm;	// Estimating effect of Ar prior
			std::cout << "ln(Z) = " << evidence << std::endl;// << std::endl;
			//}
			
			break;
		}
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

// Sample both dwarfs and giants
// N_threads	 # of parallel affine samplers to run
bool sample_affine_both(TModel &model, MCMCParams &p, TStellarData::TMagnitudes &mag, TMultiBinner<4> &multibinner, TStats &stats, double &evidence, unsigned int N_samplers=100, unsigned int N_steps=15000, unsigned int N_threads=4)
{
	unsigned int size = N_samplers;		// # of chains in each affine sampler
	unsigned int max_rounds = 5;		// After <max_rounds> rounds, the Markov chains are terminated
	double convergence_threshold = 1.1;	// Chains ended when GR diagnostic falls below this level
	double nonconvergence_flag = 1.2;	// Return false if GR diagnostic is above this level at end of run
	bool convergence[2];
	
	timespec t_start, t_end;
	clock_gettime(CLOCK_REALTIME, &t_start); // Start timer
	
	// Set run parameters
	p.update(mag);
	TNullLogger nulllogger;
	TAffineSampler<MCMCParams, TNullLogger >::pdf_t pdf_ptr = &calc_logP;
	TAffineSampler<MCMCParams, TNullLogger >::rand_state_t rand_state_ptr = &ran_state;
	
	unsigned int count, tmp_max_rounds;
	double highest_GR, tmp_GR;
	
	TChain chain(4, 2*N_steps*N_samplers);
	
	// Sample the dwarf posterior, then the giant posterior
	for(unsigned int giant_flag=1; giant_flag<3; giant_flag++) {
		p.giant_flag = giant_flag;
		TParallelAffineSampler<MCMCParams, TNullLogger > sampler(pdf_ptr, rand_state_ptr, 4, size, p, nulllogger, N_threads);
		
		// Burn-in
		sampler.set_scale(5.);
		sampler.step(N_steps, false, 4.);
		sampler.clear();
		
		// Main run
		sampler.set_scale(3.);
		count = 0;
		convergence[giant_flag-1] = false;
		while((count < max_rounds) && !convergence[giant_flag-1]) {
			sampler.step(N_steps, true, 0);
			highest_GR = -1.;
			for(unsigned int i=0; i<4; i++) {
				tmp_GR = sampler.get_GR_diagnostic(i);
				if(tmp_GR > highest_GR) { highest_GR = tmp_GR; }
			}
			if(highest_GR < convergence_threshold) { convergence[giant_flag-1] = true; }
			count++;
		}
		
		//sampler.print_stats();
		
		TChain tmp_chain = sampler.get_chain();
		if(giant_flag == 1) {
			chain.append(tmp_chain, false);		// Log dwarf solution
		} else {
			evidence = chain.append(tmp_chain, true, true, 10., 0.1, 0.1);	// Attach giant solution to dwarf solution, weighting each according to evidence
		}
		
		if(!convergence[giant_flag-1]) { std::cout << (giant_flag == 1 ? "Dwarfs" : "Giants") << " did not converge." << std::endl; }
		std::cout << std::endl;
	}
	
	// Calculate evidence
	double L_norm = 0.;
	for(unsigned int i=0; i<NBANDS; i++) {
		if(mag.err[i] < 1.e9) {
			L_norm += 0.918938533 + log(mag.err[i]);
		}
	}
	//evidence = chain.get_ln_Z_harmonic(true, 1e6, 0.1, 0.1);
	evidence -= 2.3 + L_norm;	// Estimating effect of Ar prior
	std::cout << "ln(Z) = " << evidence << std::endl << std::endl;
	
	// Calculate likelihood of max. likelihood point
	/*double peak[4];
	chain.density_peak(&(peak[0]) , 0.05);
	double M[NBANDS];
	FOR(0, NBANDS) { M[i] = p.m[i] - peak[_DM] - peak[_Ar]*p.model.Acoef[i]; }
	TSED sed_bilin_interp = (*p.model.sed_interp)(peak[_Mr], peak[_FeH]);
	double logL_peak = logL_SED(M, p.err, sed_bilin_interp) - L_norm;
	std::cout << "ln(L_peak) = " << logL_peak << std::endl << std::endl;*/
	
	// Log the results
	stats = chain.stats;
	for(unsigned int i=0; i<chain.get_length(); i++) {
		multibinner(chain[i], 1.e10*chain.get_w(i));
	}
	
	// Normalize bins to peak value
	for(unsigned int i=0; i<multibinner.get_num_binners(); i++) {
		multibinner.get_binner(i)->normalize();
	}
	
	clock_gettime(CLOCK_REALTIME, &t_end);	// End timer
	
	// Print stats and run time
	stats.print();
	if((!convergence[0]) || (!convergence[1])) { std::cout << std::endl << "Did not converge." << std::endl; }
	std::cout << std::endl << "Time elapsed for " << chain.get_length() << " samples on " << N_threads << " threads: " << std::setprecision(3) << (double)(t_end.tv_sec-t_start.tv_sec + (t_end.tv_nsec - t_start.tv_nsec)/1e9) << " s" << std::endl << std::endl;
	
	return (convergence[0] && convergence[1]);
}
