#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       fit_pdfs.py
#       
#       Copyright 2012 Greg <greg@greg-G53JW>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#       

import sys, argparse
from os.path import abspath, exists
from time import time

import numpy as np
import scipy.ndimage.filters as filters
from scipy import weave
import scipy.optimize as opt

import matplotlib as mplib
import matplotlib.pyplot as plt

from galstar_io import *
from galstarutils import get_objects



#
# OPTIMIZATION ROUTINES
#

# Compute the line integral through multiple images, stacked in <img>
def line_integral(Delta_y, img):
	# Determine the number of bins per piecewise linear region
	if img.shape[1] % Delta_y.shape[0] != 0:
		raise Exception('Number of samples in mu (%d) not integer multiple of number of piecewise linear regions (%d).' % (img.shape[1], Delta_y.shape[0]))
	N_images = img.shape[0]
	y_max = img.shape[2]
	N_regions = Delta_y.shape[0]
	N_samples = img.shape[1] / N_regions
	
	line_int_ret = np.zeros(N_images)
	code = """
		double y = 0.;
		double y_ceil, y_floor;
		int x = 0;
		for(int i=0; i<N_regions; i++) {
			//double dy = (y_anchors(i+1) - y_anchors(i)) / (double)N_samples;
			double dy = (double)(Delta_y(i)) / (double)N_samples;
			for(int j=0; j<N_samples; j++, x++) {
				y += dy;
				if(y > y_max - 1) { continue; }
				y_ceil = ceil(y);
				y_floor = floor(y);
				for(int k=0; k<N_images; k++) {
					line_int_ret(k) += (y_ceil - y) * img(k, x, (int)y_floor) + (y - y_floor) * img(k, x, (int)y_ceil);
				}
			}
			if(y > y_max - 1) { continue; }
		}
	"""
	weave.inline(code, ['img', 'Delta_y', 'N_images', 'N_regions', 'N_samples', 'y_max', 'line_int_ret'], type_converters=weave.converters.blitz, compiler='gcc')
	
	return line_int_ret


# Return chi for the model with steps in reddening given by <log_Delta_y>
def chi_leastsq(log_Delta_y, pdfs=None, chimax=5., regulator=10000.):
	Delta_y = np.exp(log_Delta_y)
	
	chi2_tmp = -2. * np.log(line_integral(Delta_y, pdfs))
	chi2scaled = chimax*chimax * np.tanh(chi2_tmp / (chimax*chimax))
	
	chi2scaled += np.sum(Delta_y*Delta_y) / (regulator*regulator)
	#chi2scaled += np.sum(log_Delta_y*log_Delta_y) / (regulator*regulator)
	
	return np.sqrt(chi2scaled)


# Minimize chi^2 for a line running through the given pdfs
def min_leastsq(pdfs, N_regions=15, chimax=5., regulator=10000.):
	# Generate a guess, based on the stacked pdfs
	pdf_stacked = np.average(pdfs, axis=0).T
	pdf_stacked /= np.max(pdf_stacked, axis=0)
	pdf_stacked.shape = (1, pdfs.shape[1], pdfs.shape[2])
	guess = np.log(np.random.ranf(N_regions) * 2.*float(pdfs.shape[2])/float(N_regions))	# Zeroeth-order guess
	guess = opt.fmin(chi_leastsq, guess, args=(pdf_stacked, chimax, regulator), ftol=1.e-6, maxiter=100000, maxfun=1e8)	# A better guess
	print 'guess:', np.exp(guess)
	chi = chi_leastsq(guess, pdfs, chimax=5.)
	print 'chi^2:', np.sum(chi*chi)
	
	# Do the full fit
	x, success = opt.leastsq(chi_leastsq, guess, args=(pdfs, chimax, regulator), ftol=1.e-6, maxfev=10000)
	measure = chi_leastsq(x, pdfs, chimax, regulator)
	
	return x, success, guess, measure


# Return a measure to minimize by simulated annealing
def anneal_measure(log_Delta_y, pdfs, p0=1.e-4, regulator=10000.):
	Delta_y = np.exp(log_Delta_y)
	if np.any(np.isnan(Delta_y)):
		raise ValueError('Delta_y contains NaN values.')
	
	measure = line_integral(Delta_y, pdfs)	# Begin with line integral through each stellar pdf
	
	measure += p0 * np.exp(-measure/p0)						# Soften around zero (measure -> positive const. below scale p0)
	#measure = p0 * np.log(2. * np.cosh(measure / p0))
	measure = -np.sum(np.log(measure))					# Sum logarithms of line integrals
	
	# Disfavor larger values of Delta_y slightly
	#measure += np.sum(Delta_y*Delta_y) / (regulator*regulator)
	
	# Disfavor larger values of ln(Delta_y) slightly
	measure += np.sum(log_Delta_y*log_Delta_y) / (2.*regulator*regulator)
	
	#print measure
	return measure


# Maximize the line integral by simulated annealing
def min_anneal(pdfs, N_regions=15, p0=1.e-5, regulator=10000., dwell=1000):
	# Start with random guess
	guess = np.log(0.5 * np.random.ranf(N_regions) * 2.* float(pdfs.shape[2])/float(N_regions))
	
	# Set bounds on step size in Delta_Ar
	lower = np.empty(N_regions, dtype=np.float64)
	upper = np.empty(N_regions, dtype=np.float64)
	lower.fill(-0.02)
	upper.fill(0.02)
	
	# Run simulated annealing
	#feps=1.e-12
	x, success = opt.anneal(anneal_measure, guess, args=(pdfs, p0, regulator), lower=lower, upper=upper, maxiter=1000, dwell=dwell)
	measure = anneal_measure(x, pdfs, p0, regulator)
	
	return x, success, guess, measure


# Fit line-of-sight reddening profile, given the binned pdfs in <bin_fname> and stats in <stats_fname>
def fit_los(bin_fname, stats_fname, N_regions, sparse=True, converged=False, method='anneal', smooth=(1,1), regulator=10000., dwell=1000):
	# Load pdfs
	sys.stderr.write('Loading binned pdfs...\n')
	bounds, p = None, None
	bounds, p = load_bins(bin_fname, sparse)
	mask = np.logical_not(np.sum(np.sum(np.logical_not(np.isfinite(p)), axis=1), axis=1).astype(np.bool))	# Filter out images with NaN bins
	if converged:	# Filter out nonconverged images
		converged, means, cov = load_stats(stats_fname)
		mask = np.logical_and(mask, converged)
		p = smooth_bins(p[mask], smooth)
	else:
		p = smooth_bins(p[mask], smooth)
	
	# Fit reddening profile
	x, success, guess, measure = None, None, None, None
	if method == 'leastsq':
		sys.stderr.write('Fitting reddening profile using the LM method (scipy.optimize.leastsq)...\n')
		x, success, guess, measure = min_leastsq(p, N_regions=N_regions, chimax=5., regulator=regulator)
	elif method == 'anneal':
		sys.stderr.write('Fitting reddening profile using simulated annealing (scipy.optimize.anneal)...\n')
		x, success, guess, measure = min_anneal(p, N_regions=N_regions, p0=1.e-5, regulator=regulator, dwell=dwell)
	
	# Convert output into physical coordinates (rather than pixel coordinates)
	Delta_Ar = np.exp(x) * ((bounds[3] - bounds[2]) / float(p.shape[2]))
	guess = np.exp(guess) * ((bounds[3] - bounds[2]) / float(p.shape[2]))
	
	# Output basic information about fit
	sys.stderr.write('Delta_Ar: %s\n' % np.array_str(Delta_Ar, max_line_width=N_regions*100, precision=8))
	sys.stderr.write('success: %d\n' % success)
	sys.stderr.write('measure: %f\n' % measure)
	
	return bounds, p, measure, success, Delta_Ar, guess




#
# PLOTS
#

# Overplot reddening profile on stacked pdfs
def plot_profile(bounds, p, Delta_Ar, plot_fn=None, overplot=None):
	# Calculate reddening profile
	N_regions = Delta_Ar.size
	mu_anchors = np.linspace(bounds[0], bounds[1], N_regions+1)
	Ar_anchors = np.empty(N_regions+1, dtype=Delta_Ar.dtype)
	for i in xrange(N_regions+1):
		Ar_anchors[i] = bounds[2] + np.sum(Delta_Ar[:i])
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# Make figure
	fig = plt.figure(figsize=(7,5), dpi=100)
	ax = fig.add_subplot(1,1,1)
	
	# Plot the stacked pdfs
	img = np.average(p, axis=0).T
	img /= np.max(img, axis=0)
	img.shape = (1, p.shape[2], p.shape[1])
	ax.imshow(img[0], extent=bounds, origin='lower', aspect='auto', cmap='hot')
	
	# Overplot locations of stars from galfast
	if overplot != None:
		# Load the true positions of the stars to overlplot
		ra_dec, mags, errs, params = get_objects(abspath(overplot))
		x = params[:,0]
		y = params[:,1]
		ax.plot(x, y, 'g.', linestyle='None', markersize=2, alpha=0.3)
	
	# Plot the line-of-sight reddening profile
	ax.plot(mu_anchors, Ar_anchors)
	
	# Set axis limits and labels
	y_max = min([bounds[3], 2.*np.max(Ar_anchors)])
	ax.set_xlim(bounds[0], bounds[1])
	ax.set_ylim(bounds[2], y_max)
	ax.set_xlabel(r'$\mu$', fontsize=18)
	ax.set_ylabel(r'$A_r$', fontsize=18)
	fig.subplots_adjust(bottom=0.10)
	
	if plot_fn != None:
		fig.savefig(abspath(plot_fn), dpi=150)


def output_profile(fname, pixnum, bounds, Delta_Ar):
	'''
	Append the reddening profile to the end of the binary file given by <fname>.
	
	Format - for each pixel:
		pixnum		(uint64)
		N_regions	(uint16)
		mu_anchors	(float64)
		Ar_anchors	(float64)
	'''
	# Calculate reddening profile
	N_regions = Delta_Ar.size
	mu_anchors = np.linspace(bounds[0], bounds[1], N_regions+1)
	Ar_anchors = np.empty(N_regions+1, dtype=Delta_Ar.dtype)
	for i in xrange(N_regions+1):
		Ar_anchors[i] = bounds[2] + np.sum(Delta_Ar[:i])
	
	f = open(fname, 'wb')
	f.seek(0, 2)	# Seek to end of file
	f.write(np.array([pixnum], dtype=np.uint64).tostring())
	f.write(np.array([N_regions], dtype=np.uint16).tostring())
	f.write(mu_anchors.tostring())
	f.write(Ar_anchors.tostring())
	f.close()




#
# MAIN
#

def main():
	parser = argparse.ArgumentParser(prog='fit_pdfs.py', description='Fit line-of-sight reddening law from probability density functions of individual stars.', add_help=True)
	parser.add_argument('binfn', type=str, help='File containing binned probability density functions for each star along l.o.s. (also accepts gzipped files)')
	parser.add_argument('statsfn', type=str, help='File containing summary statistics for each star.')
	parser.add_argument('-N', '--N', type=int, default=15, help='# of piecewise-linear regions in DM-Ar relation')
	parser.add_argument('-mtd', '--method', type=str, choices=('anneal', 'leastsq'), default='anneal', help='Optimization method (default: anneal)')
	parser.add_argument('-cnv', '--converged', action='store_true', help='Filter out unconverged stars.')
	parser.add_argument('-sm', '--smooth', type=int, nargs=2, default=(1,1), help='Std. dev. of smoothing kernel (in pixels) for individual pdfs (default: 1 1).')
	parser.add_argument('-reg', '--regulator', type=float, default=10000., help='Width of support of prior on ln(Delta_Ar) (default: 10000).')
	parser.add_argument('-o', '--outfn', type=str, nargs=2, default=None, help='Output filename for reddening profile and healpix pixel number.')
	parser.add_argument('-po', '--plotfn', type=str, default=None, help='Filename for plot of result.')
	parser.add_argument('-sh', '--show', action='store_true', help='Show plot of result.')
	parser.add_argument('-ovp', '--overplot', type=str, default=None, help='Overplot true values from galfast FITS file')
	parser.add_argument('-dw', '--dwell', type=int, default=1000, help='dwell parameter for annealing algorithm. The higher the value, the greater the chance of convergence (default: 1000).')
	parser.add_argument('-nsp', '--nonsparse', action='store_true', help='Binned pdfs are not stored in sparse format.')
	#parser.add_argument('-v', '--verbose', action='store_true', help='Print information on fit.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	np.seterr(all='ignore')
	
	tstart = time()
	
	# Fit the line of sight
	bounds, p, measure, success, Delta_Ar, guess = fit_los(values.binfn, values.statsfn, values.N, sparse=(not values.nonsparse), converged=values.converged, method=values.method, smooth=values.smooth, regulator=values.regulator, dwell=values.dwell)
	duration = time() - tstart
	sys.stderr.write('Time elapsed: %.1f s\n' % duration)
	
	# Save the reddening profile to an ASCII file, or print to stdout
	output_profile(values.outfn[0], int(values.outfn[1]), bounds, Delta_Ar)
	
	# Plot the reddening profile on top of the stacked stellar probability densities
	if (values.plotfn != None) or values.show:
		sys.stderr.write('Plotting profile to %s ...' % values.plotfn)
		plot_profile(bounds, p, Delta_Ar, values.plotfn, values.overplot)
	
	if values.show:
		plt.show()
	
	return 0


if __name__ == '__main__':
	main()

