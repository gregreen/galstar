
#include "chain.h"

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

void TChain::append(const TChain& chain, bool reweight, double nsigma) {
	assert(chain.N == N);	// Make sure the two chains have the same dimensionality
	
	// Weight each chain according to Bayesian evidence, if requested
	double a = 1.;
	if(reweight) {
		a = chain.get_Z_harmonic(nsigma) / get_Z_harmonic(nsigma) * total_weight / chain.total_weight;
	}
	
	// Append the last chain to this one
	if(capacity < length + chain.length) { set_capacity(1.5*(length + chain.length)); }
	std::vector<double>::iterator w_end_old = w.end();
	x.insert(x.end(), chain.x.begin(), chain.x.end());
	L.insert(L.end(), chain.L.begin(), chain.L.end());
	w.insert(w.end(), chain.w.begin(), chain.w.end());
	
	if(reweight) {
		std::cout << "a = " << a << std::endl;
		std::vector<double>::iterator w_end = w.end();
		for(std::vector<double>::iterator it = w_end_old; it != w_end; ++it) {
			*it *= a;
		}
	}
	
	stats += a * chain.stats;
	length += chain.length;
	total_weight += a * chain.total_weight;
	
	//stats.clear();
	//for(unsigned int i=0; i<length; i++) {
	//	stats(get_element(i), 1.e10*w[i]);
	//}
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
	
	bool stats_success = stats.write_binary(filename.c_str(), std::ios::app);
	
	return stats_success;
}


bool TChain::load(std::string filename, bool reserve_extra){
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
	}
	return *this;
}


// Uses a variant of the harmonic mean approximation to determine the evidence.
// Essentially, the regulator chosen is an ellipsoid with radius nsigma standard deviations
// along each principal axis. The regulator is then 1/V inside the ellipsoid and 0 without,
// where V is the volume of the ellipsoid. In this form, the harmonic mean approximation
// has finite variance. See Gelfand & Dey (1994) and Robert & Wraith (2009) for details.
double TChain::get_Z_harmonic(double nsigma) const {
	// Get the covariance and determinant of the chain
	gsl_matrix* Sigma = gsl_matrix_alloc(N, N);
	gsl_matrix* invSigma = gsl_matrix_alloc(N, N);
	double detSigma;
	stats.get_cov_matrix(Sigma, invSigma, &detSigma);
	
	// Get the mean from the stats class
	double* mu = new double[N];
	for(unsigned int i=0; i<N; i++) {
		mu[i] = stats.mean(i);
	}
	
	// Determine the volume normalization (the prior volume)
	double V = sqrt(detSigma) * 2. * pow(SQRTPI * nsigma, (double)N) / (double)(N) / gsl_sf_gamma((double)(N)/2.);
	
	// Determine <1/L> inside the prior volume
	double dist2;
	double sum_invL = 0;
	for(unsigned int i=0; i<length; i++) {
		dist2 = metric_dist2(invSigma, get_element(i), mu, N);
		if(dist2 < nsigma*nsigma) {
			sum_invL += w[i] / exp(L[i]);
		}
	}
	
	// Cleanup
	gsl_matrix_free(Sigma);
	gsl_matrix_free(invSigma);
	delete[] mu;
	
	// Return the estimate of Z
	return V / sum_invL * total_weight;
}
