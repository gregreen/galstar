#ifndef _marginalize_h__
#define _marginalize_h__

#include "sampler.h"
#include <map>
#include <iostream>

#include <boost/format.hpp>
#include <boost/array.hpp>
#include <boost/static_assert.hpp>
/*
	Marginalize the samples over all but two parameters
*/

struct IMarginalizer
{
	virtual void output(std::ostream &out) const = 0;

	virtual void normalize() = 0;
	virtual void normalize_to_peak() = 0;
};

#define call_memfun(object, ptrToMember)  ((object).*(ptrToMember))
template<int DIM>
struct Marginalizer : public TModel::Marginalizer, public IMarginalizer
{
public:
	typedef TModel::Params::Getter Getter;

	Getter get[DIM];
	mutable boost::format fmt;

public:
	Marginalizer(const std::string &fmt_string = "%5.2f % 8.3f\n")
		: fmt(fmt_string)
	{
		BOOST_STATIC_ASSERT(DIM == 0);
	}
	Marginalizer(const Getter X, const std::string &fmt_string = "%5.2f % 8.3f\n")
		: fmt(fmt_string)
	{
		BOOST_STATIC_ASSERT(DIM == 1);
		get[0] = X;
	}
	Marginalizer(const Getter X, const Getter Y, const std::string &fmt_string = "%5.2f %5.2f % 8.3f\n")
		: fmt(fmt_string)
	{
		BOOST_STATIC_ASSERT(DIM == 2);
		get[0] = X; get[1] = Y;
	}
	Marginalizer(const Getter X, const Getter Y, const Getter Z, const std::string &fmt_string = "%5.2f %5.2f %5.2f % 8.3f\n")
		: fmt(fmt_string)
	{
		BOOST_STATIC_ASSERT(DIM == 3);
		get[0] = X; get[1] = Y; get[1] = Z;
	}

	std::map<boost::array<double, DIM>, double> posterior;

	virtual void operator()(const TModel::Params &p, double logP)
	{
		boost::array<double, DIM> key;

		if(DIM > 0) { key[0] = call_memfun(p, get[0])(); }
		if(DIM > 1) { key[1] = call_memfun(p, get[1])(); }
		if(DIM > 2) { key[2] = call_memfun(p, get[2])(); }

		posterior[key] += (logP);
	}

	virtual void output(std::ostream &out) const
	{
		// ASCII column output: [X [Y [Z]]] logP
		FOREACH(posterior)
		{
			double logP = log(i->second);
			if(logP < -1e4) { continue; }

			if(DIM == 1) out << fmt % i->first[0] % logP;
			if(DIM == 2) out << fmt % i->first[0] % i->first[1] % logP;
			if(DIM == 3) out << fmt % i->first[0] % i->first[1] % i->first[2] % logP;
		}
	}

	// Normalize the posterior pdf
	virtual void normalize()
	{
		double sum = 0;
		FOREACH(posterior) { sum += i->second; }
		FOREACH(posterior) { i->second /= sum; }
	}

	// Normalize the posterior pdf
	virtual void normalize_to_peak()
	{
		double peak = posterior.empty() ? 1 : posterior.begin()->second;
		FOREACH(posterior) { peak = std::max(peak, i->second); }
		FOREACH(posterior) { i->second /= peak; }
	}
};
template<int DIM>
inline std::ostream &operator <<(std::ostream &out, const Marginalizer<DIM> &m)
{
	m.output(out);
	return out;
}

#endif // _marginalize_h__