
#include "chain.h"
#include <string.h>

/*************************************************************************
 *   Chain Class Member Functions
 *************************************************************************/

// Standard constructor
TChain::TChain(unsigned int _N, unsigned int _capacity)
	: stats(_N)
{
	N = _N;
	length = 0;
	total_weight = 0;
	set_capacity(_capacity);
	
	// Initialize min/max coordinates
	x_min.reserve(N);
	x_max.reserve(N);
	for(unsigned int i=0; i<N; i++) {
		x_min.push_back(std::numeric_limits<double>::infinity());
		x_max.push_back(-std::numeric_limits<double>::infinity());
	}
}

// Copy constructor
TChain::TChain(const TChain& c)
	: stats(1)
{
	stats = c.stats;
	x = c.x;
	L = c.L;
	w = c.w;
	total_weight = c.total_weight;
	N = c.N;
	length = c.length;
	capacity = c.capacity;
	x_min = c.x_min;
	x_max = c.x_max;
}

// Construct the string from file
TChain::TChain(std::string filename, bool reserve_extra)
	: stats(1)
{
	bool load_success = load(filename, reserve_extra);
	if(!load_success) {
		std::cout << "Failed to load " << filename << " into chain." << std::endl;
	}
}

TChain::~TChain() {}

void TChain::add_point(double* element, double L_i, double w_i) {
	stats(element, (unsigned int)w_i);
	for(unsigned int i=0; i<N; i++) {
		x.push_back(element[i]);
		if(element[i] < x_min[i]) { x_min[i] = element[i]; }
		if(element[i] > x_max[i]) { x_max[i] = element[i]; }
	}
	L.push_back(L_i);
	w.push_back(w_i);
	total_weight += w_i;
	length += 1;
}

void TChain::clear() {
	x.clear();
	L.clear();
	w.clear();
	stats.clear();
	total_weight = 0;
	length = 0;
	
	// Reset min/max coordinates
	for(unsigned int i=0; i<N; i++) {
		x_min[i] = std::numeric_limits<double>::infinity();
		x_max[i] = -std::numeric_limits<double>::infinity();
	}
}

void TChain::set_capacity(unsigned int _capacity) {
	capacity = _capacity;
	x.reserve(N*capacity);
	L.reserve(capacity);
	w.reserve(capacity);
}

unsigned int TChain::get_capacity() const {
	return capacity;
}

unsigned int TChain::get_length() const {
	return length;
}

double TChain::get_total_weight() const {
	return total_weight;
}

const double* TChain::get_element(unsigned int i) const {
	return &(x[i*N]);
}

double TChain::get_L(unsigned int i) const {
	return L[i];
}

double TChain::get_w(unsigned int i) const {
	return w[i];
}

double TChain::append(const TChain& chain, bool reweight, bool use_peak, double nsigma_max, double nsigma_peak, double chain_frac, double threshold) {
	assert(chain.N == N);	// Make sure the two chains have the same dimensionality
	
	// Weight each chain according to Bayesian evidence, if requested
	double a1 = 1.;
	double a2 = 1.;
	double lnZ = 0.;
	if(reweight) {
		double lnZ1 = chain.get_ln_Z_harmonic(use_peak, nsigma_max, nsigma_peak, chain_frac);
		double lnZ2 = get_ln_Z_harmonic(use_peak, nsigma_max, nsigma_peak, chain_frac);
		
		if(lnZ2 > lnZ1) {
			a2 = exp(lnZ1 - lnZ2) * total_weight / chain.total_weight;
			/*if(isnan(a2)) {
				std::cout << std::endl;
				std::cout << "NaN Error: a2 = " << a2 << std::endl;
				std::cout << "ln(Z1) = " << lnZ1 << std::endl;
				std::cout << "ln(Z2) = " << lnZ2 << std::endl;
				std::cout << "total_weight = " << total_weight << std::endl;
				std::cout << "chain.total_weight = " << chain.total_weight << std::endl;
				std::cout << std::endl;
			}*/
		} else {
			a1 = exp(lnZ2 - lnZ1) * chain.total_weight / total_weight;
			/*if(isnan(a1)) {
				std::cout << std::endl;
				std::cout << "NaN Error: a1 = " << a1 << std::endl;
				std::cout << "ln(Z1) = " << lnZ1 << std::endl;
				std::cout << "ln(Z2) = " << lnZ2 << std::endl;
				std::cout << "total_weight = " << total_weight << std::endl;
				std::cout << "chain.total_weight = " << chain.total_weight << std::endl;
				std::cout << std::endl;
			}*/
		}
		
		lnZ = log(a1/(a1+a2) * exp(lnZ2) + a2/(a1+a2) * exp(lnZ1));
	}
	
	if(reweight) { std::cout << "(a1, a2) = " << a1 << ", " << a2 << std::endl; }
	
	// Append the last chain to this one
	if(reweight && (a1 < threshold)) {
		x = chain.x;
		L = chain.L;
		w = chain.w;
		length = chain.length;
		capacity = chain.capacity;
		stats = chain.stats;
		total_weight = chain.total_weight;
		for(unsigned int i=0; i<N; i++) {
			x_max[i] = chain.x_max[i];
			x_min[i] = chain.x_min[i];
		}
	} else if(!(reweight && (a2 < threshold))) {
		if(capacity < length + chain.length) { set_capacity(1.5*(length + chain.length)); }
		std::vector<double>::iterator w_end_old = w.end();
		x.insert(x.end(), chain.x.begin(), chain.x.end());
		L.insert(L.end(), chain.L.begin(), chain.L.end());
		w.insert(w.end(), chain.w.begin(), chain.w.end());
		
		if(reweight) {
			std::vector<double>::iterator w_end = w.end();
			for(std::vector<double>::iterator it = w.begin(); it != w_end_old; ++it) {
				*it *= a1;
			}
			for(std::vector<double>::iterator it = w_end_old; it != w_end; ++it) {
				*it *= a2;
			}
		}
		
		stats *= a1;
		stats += a2 * chain.stats;
		length += chain.length;
		total_weight *= a1;
		total_weight += a2 * chain.total_weight;
		
		// Update min/max coordinates
		for(unsigned int i=0; i<N; i++) {
			if(chain.x_max[i] > x_max[i]) { x_max[i] = chain.x_max[i]; }
			if(chain.x_min[i] < x_min[i]) { x_min[i] = chain.x_min[i]; }
		}
	}
	//stats.clear();
	//for(unsigned int i=0; i<length; i++) {
	//	stats(get_element(i), 1.e10*w[i]);
	//}
	
	return lnZ;
}

double* TChain::operator [](unsigned int i) {
	return &(x[i*N]);
}

void TChain::operator +=(const TChain& rhs) {
	append(rhs);
}

TChain& TChain::operator =(const TChain& rhs) {
	if(&rhs != this) {
		stats = rhs.stats;
		x = rhs.x;
		L = rhs.L;
		w = rhs.w;
		total_weight = rhs.total_weight;
		N = rhs.N;
		length = rhs.length;
		capacity = rhs.capacity;
		x_min = rhs.x_min;
		x_max = rhs.x_max;
	}
	return *this;
}


// A structure used to sort the elements of the chain
struct TChainSort {
	unsigned int index;
	double dist2;
	
	bool operator<(const TChainSort rhs) const { return dist2 < rhs.dist2; }	// Reversed so that the sort is in ascending order
};

//bool chainsortcomp(TChainSort a, TChainSort b) { return a.dist2 < b.dist2; }


// Uses a variant of the bounded harmonic mean approximation to determine the evidence.
// Essentially, the regulator chosen is an ellipsoid with radius nsigma standard deviations
// along each principal axis. The regulator is then 1/V inside the ellipsoid and 0 without,
// where V is the volume of the ellipsoid. In this form, the harmonic mean approximation
// has finite variance. See Gelfand & Dey (1994) and Robert & Wraith (2009) for details.
double TChain::get_ln_Z_harmonic(bool use_peak, double nsigma_max, double nsigma_peak, double chain_frac) const {
	// Get the covariance and determinant of the chain
	gsl_matrix* Sigma = gsl_matrix_alloc(N, N);
	gsl_matrix* invSigma = gsl_matrix_alloc(N, N);
	double detSigma;
	stats.get_cov_matrix(Sigma, invSigma, &detSigma);
	
	//std::cout << std::endl << use_peak << "\t" << nsigma_max << "\t" << nsigma_peak << "\t" << chain_frac << std::endl;
	
	/*std::cout << "Covariance:" << std::endl;
	for(unsigned int i=0; i<N; i++) {
		std::cout << std::endl;
		for(unsigned int j=0; j<N; j++) { std::cout << gsl_matrix_get(Sigma, i, j) << "\t"; }
	}
	std::cout << std::endl << std::endl;*/
	
	// Determine the center of the prior volume to use
	double* mu = new double[N];
	if(use_peak) {	// Use the peak density as the center
		find_center(mu, Sigma, invSigma, &detSigma, nsigma_peak, 5);
		//density_peak(mu, nsigma_peak);
	} else {	// Get the mean from the stats class
		for(unsigned int i=0; i<N; i++) { mu[i] = stats.mean(i); }
	}
	//std::cout << std::endl << "mu = ";
	//for(unsigned int i=0; i<N; i++) { std::cout << mu[i] << "\t"; }
	//std::cout << std::endl;
	
	// Sort elements in chain by distance from center, filtering out values of L which are not finite
	std::vector<TChainSort> sorted_indices;
	sorted_indices.reserve(length);
	unsigned int filt_length = 0;
	for(unsigned int i=0; i<length; i++) {
		if(!(isnan(L[i]) || isinf(L[i]))) {
			TChainSort tmp_el;
			tmp_el.index = i;
			tmp_el.dist2 = metric_dist2(invSigma, get_element(i), mu, N);
			sorted_indices.push_back(tmp_el);
			filt_length++;
		}
	}
	unsigned int npoints = (unsigned int)(chain_frac * (double)filt_length);
	std::partial_sort(sorted_indices.begin(), sorted_indices.begin() + npoints, sorted_indices.end());
	/*for(unsigned int i=0; i<20; i++) {
		std::cout << sorted_indices[i].index << "\t" << sorted_indices[i].dist2 << std::endl;
	}*/
	//std::cout << "filtered: " << length - filt_length << std::endl;
	
	// Determine <1/L> inside the prior volume
	double sum_invL = 0.;
	double tmp_invL;
	double nsigma = sqrt(sorted_indices[npoints-1].dist2);
	unsigned int tmp_index = sorted_indices[0].index;;
	double L_0 = L[tmp_index];
	//std::cout << "index_0 = " << sorted_indices[0].index << std::endl;
	for(unsigned int i=0; i<npoints; i++) {
		if(sorted_indices[i].dist2 > nsigma_max * nsigma_max) {
			//std::cout << "Counted only " << i << " elements in chain." << std::endl;
			nsigma = nsigma_max;
			break;
		}
		tmp_index = sorted_indices[i].index;
		//std::cout << "\t" << tmp_index << "\t" << L[tmp_index] << std::endl;
		tmp_invL = w[tmp_index] / exp(L[tmp_index] - L_0);
		if(isnan(tmp_invL)) {
			std::cout << "\t\tL, L_0 = " << L[tmp_index] << ", " << L_0 << std::endl;
		}
		if((tmp_invL + sum_invL > 1.e100) && (i != 0)) {
			nsigma = sqrt(sorted_indices[i-1].dist2);
			break;
		}
		sum_invL += tmp_invL;
	}
	//std::cout << "sum_invL = e^(" << -L_0 << ") * " << sum_invL << " = " << exp(-L_0) * sum_invL << std::endl;
	//std::cout << "nsigma = " << nsigma << std::endl;
	
	// Determine the volume normalization (the prior volume)
	double V = sqrt(detSigma) * 2. * pow(SQRTPI * nsigma, (double)N) / (double)(N) / gsl_sf_gamma((double)(N)/2.);
	//std::cout << "V = " << V << std::endl;
	//std::cout << "total_weight = " << total_weight << std::endl << std::endl;
	
	// Return an estimate of ln(Z)
	double lnZ = log(V) - log(sum_invL) + log(total_weight) + L_0;
	
	if(isnan(lnZ)) {
		std::cout << std::endl;
		std::cout << "NaN Error! lnZ = " << lnZ << std::endl;
		std::cout << "\tsum_invL = e^(" << -L_0 << ") * " << sum_invL << " = " << exp(-L_0) * sum_invL << std::endl;
		std::cout << "\tV = " << V << std::endl;
		std::cout << "\ttotal_weight = " << total_weight << std::endl;
		std::cout << std::endl;
	} else if(isinf(lnZ)) {
		std::cout << std::endl;
		std::cout << "inf Error! lnZ = " << lnZ << std::endl;
		std::cout << "\tsum_invL = e^(" << -L_0 << ") * " << sum_invL << " = " << exp(-L_0) * sum_invL << std::endl;
		std::cout << "\tV = " << V << std::endl;
		std::cout << "\ttotal_weight = " << total_weight << std::endl;
		std::cout << "\tnsigma = " << nsigma << std::endl;
		std::cout << "\tIndex\tDist^2:" << std::endl;
		for(unsigned int i=0; i<10; i++) {
			std::cout << sorted_indices[i].index << "\t\t" << sorted_indices[i].dist2 << std::endl;
			std::cout << "  ";
			const double *tmp_x = get_element(sorted_indices[i].index);
			for(unsigned int k=0; k<N; k++) { std::cout << " " << tmp_x[k]; }
			std::cout << std::endl;
		}
		std::cout << "mu =";
		for(unsigned int i=0; i<N; i++) { std::cout << " " << mu[i]; }
		std::cout << std::endl;
	}
	
	//std::cout << "mu =";
	//for(unsigned int i=0; i<N; i++) { std::cout << " " << mu[i]; }
	//std::cout << std::endl;
	
	// Cleanup
	gsl_matrix_free(Sigma);
	gsl_matrix_free(invSigma);
	delete[] mu;
	
	return lnZ;
	//return V / sum_invL * total_weight;
}


// Estimate the coordinate with peak density.
void TChain::density_peak(double* const peak, double nsigma) const {
	// Width of bin in each direction
	double* width = new double[N];
	uint64_t* index_width = new uint64_t[N];
	uint64_t* mult = new uint64_t[N];
	mult[0] = 1;
	//std::cout << std::endl;
	for(unsigned int i=0; i<N; i++) {
		index_width[i] = (uint64_t)ceil((x_max[i] - x_min[i]) / (nsigma * sqrt(stats.cov(i,i))));
		width[i] = (x_max[i] - x_min[i]) / (double)(index_width[i]);
		//std::cout << x_min[i] << "\t" << x_max[i] << "\t" << width[i] << "\t" << index_width[i] << std::endl;
		if(i != 0) { mult[i] = mult[i-1] * index_width[i-1]; }
	}
	
	// Bin the chain
	std::map<uint64_t, double> bins;
	uint64_t index;
	std::map<uint64_t, double>::iterator it;
	for(unsigned int i=0; i<length; i++) {
		index = 0;
		for(unsigned int k=0; k<N; k++) { index += mult[k] * (uint64_t)floor((x[N*i + k] - x_min[k]) / width[k]); }
		bins[index] += w[i];
	}
	
	// Find the index of the max bin
	std::map<uint64_t, double>::iterator it_end = bins.end();
	double w_max = -std::numeric_limits<double>::infinity();
	for(it = bins.begin(); it != it_end; ++it) {
		if(it->second > w_max) {
			//std::cout << "\t" << it->second << "\t" << it->first << std::endl;
			w_max = it->second;
			index = it->first;
		}
	}
	
	// Convert the index to a coordinate
	//std::cout << index << std::endl;
	uint64_t coord_index;
	for(unsigned int i=0; i<N; i++) {
		coord_index = index % index_width[i];
		index = (index - coord_index) / index_width[i];
		//std::cout << "\t" << coord_index;
		peak[i] = x_min[i] + ((double)coord_index + 0.5) * width[i];
		//std::cout << "\t" << peak[i];
	}
	//std::cout << std::endl;
	//std::cout << index << std::endl;
	
	delete[] width;
	delete[] index_width;
	delete[] mult;
}

// Find a point in space with high density.
void TChain::find_center(double* const center, gsl_matrix *const cov, gsl_matrix *const inv_cov, double* det_cov, double dmax, unsigned int iterations) const {
	// Check that the matrices are the correct size
	/*assert(cov->size1 == N);
	assert(cov->size2 == N);
	assert(inv_cov->size1 == N);
	assert(inv_cov->size2 == N);*/
	
	// Choose random point in chain as starting point
	gsl_rng *r;
	seed_gsl_rng(&r);
	
	long unsigned int index_tmp = gsl_rng_uniform_int(r, length);
	const double *x_tmp = get_element(index_tmp);
	for(unsigned int i=0; i<N; i++) { center[i] = x_tmp[i]; }
	
	//std::cout << "center #0:";
	//for(unsigned int n=0; n<N; n++) { std::cout << " " << center[n]; }
	//std::cout << std::endl;
	
	/*double *E_k = new double[N];
	double *E_ij = new double[N*N];
	for(unsigned int n1=0; n1<N; n1++) {
		E_k[n1] = 0.;
		for(unsigned int n2=0; n2<N; n2++) { E_ij[n1 + N*n2] = 0.; }
	}*/
	
	// Iterate
	double *sum = new double[N];
	double weight;
	for(unsigned int i=0; i<iterations; i++) {
		// Set mean of nearby points as center
		weight = 0.;
		for(unsigned int n=0; n<N; n++) { sum[n] = 0.; }
		for(unsigned int k=0; k<length; k++) {
			x_tmp = get_element(k);
			if(metric_dist2(inv_cov, x_tmp, center, N) < dmax*dmax) {
				for(unsigned int n=0; n<N; n++) { sum[n] += w[k] * x_tmp[n]; }
				weight += w[k];
				
				// Calculate the covariance
				/*if(i == iterations - 1) {
					for(unsigned int n1=0; n1<N; n1++) {
						E_k[n1] += w[k] * x_tmp[n1];
						for(unsigned int n2=0; n2<N; n2++) { E_ij[n1 + N*n2] += w[k] * x_tmp[n1] * x_tmp[n2]; }
					}
				}*/
			}
		}
		//std::cout << "center #" << i+1 << ":";
		for(unsigned int n=0; n<N; n++) { center[n] = sum[n] / (double)weight; }//std::cout << " " << center[n]; }
		//std::cout << " (" << weight << ")" << std::endl;
		
		dmax *= 0.9;
	}
	
	for(unsigned int n=0; n<N; n++) { std::cout << " " << center[n]; }
	std::cout << std::endl;
	
	// Calculate the covariance matrix of the enclosed points
	/*double tmp;
	for(unsigned int i=0; i<N; i++) {
		for(unsigned int j=i; j<N; j++) {
			tmp = (E_ij[i + N*j] - E_k[i]*E_k[j]/(double)weight) / (double)weight;
			gsl_matrix_set(cov, i, j, tmp);
			if(i != j) { gsl_matrix_set(cov, j, i, tmp); }
		}
	}*/
	
	// Get the inverse of the covariance
	/*int s;
	gsl_permutation* p = gsl_permutation_alloc(N);
	gsl_matrix* LU = gsl_matrix_alloc(N, N);
	gsl_matrix_memcpy(LU, cov);
	gsl_linalg_LU_decomp(LU, p, &s);
	gsl_linalg_LU_invert(LU, p, inv_cov);
	
	// Get the determinant of the covariance
	*det_cov = gsl_linalg_LU_det(LU, s);
	
	// Cleanup
	gsl_matrix_free(LU);
	gsl_permutation_free(p);
	delete[] E_k;
	delete[] E_ij;*/
	
	gsl_rng_free(r);
	delete[] sum;
}


bool TChain::save(std::string filename) const {
	std::fstream out(filename.c_str(), std::ios::out | std::ios::binary);
	
	if(!out.good()) { return false; }
	
	out.write(reinterpret_cast<const char *>(&N), sizeof(unsigned int));
	out.write(reinterpret_cast<const char *>(&length), sizeof(unsigned int));
	out.write(reinterpret_cast<const char *>(&capacity), sizeof(unsigned int));
	out.write(reinterpret_cast<const char *>(&total_weight), sizeof(double));
	
	out.write(reinterpret_cast<const char *>(&(x[0])), N * length * sizeof(double));
	out.write(reinterpret_cast<const char *>(&(L[0])), length * sizeof(double));
	out.write(reinterpret_cast<const char *>(&(w[0])), length * sizeof(double));
	
	if(out.fail()) {
		out.close();
		return false;
	}
	
	out.close();
	
	bool stats_success = stats.write_binary_old(filename.c_str(), std::ios::app);
	
	return stats_success;
}

bool TChain::load(std::string filename, bool reserve_extra) {
	std::fstream in(filename.c_str(), std::ios::in | std::ios::binary);
	
	if(!in.good()) { return false; }
	
	in.read(reinterpret_cast<char *>(&N), sizeof(unsigned int));
	in.read(reinterpret_cast<char *>(&length), sizeof(unsigned int));
	in.read(reinterpret_cast<char *>(&capacity), sizeof(unsigned int));
	in.read(reinterpret_cast<char *>(&total_weight), sizeof(double));
	
	if(!reserve_extra) {
		capacity = length;
	}
	
	x.reserve(N*capacity);
	L.reserve(capacity);
	w.reserve(capacity);
	
	x.resize(length);
	L.resize(length);
	w.resize(length);
	
	in.read(reinterpret_cast<char *>(&(x[0])), N * length * sizeof(double));
	in.read(reinterpret_cast<char *>(&(L[0])), length * sizeof(double));
	in.read(reinterpret_cast<char *>(&(w[0])), length * sizeof(double));
	
	if(in.fail()) {
		in.close();
		return false;
	}
	
	std::streampos read_offset = in.tellg();
	in.close();
	
	bool stats_success = stats.read_binary(filename, read_offset);
	
	return stats_success;
}

