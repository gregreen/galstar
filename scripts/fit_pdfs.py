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
from os.path import abspath
import gzip
from time import time

import numpy as np
import scipy.ndimage.filters as filters
from scipy import weave
import scipy.optimize as opt

import matplotlib as mplib
import matplotlib.pyplot as plt


#
# FILE I/O
#

# Load multiple stats from one file
def load_stats(fname):
	f = open(fname, 'rb')
	
	# Read in header
	N_files = np.fromfile(f, dtype=np.uint32, count=1)[0]
	N_dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
	
	# Set up arrays to hold statistics
	converged = np.empty(N_files, dtype=np.bool)
	mean = np.empty((N_files, N_dim), dtype=np.float64)
	cov = np.empty((N_files, N_dim*N_dim), dtype=np.float64)
	
	# Read in statistics one at a time
	for i in xrange(N_files):
		converged[i] = np.fromfile(f, dtype=np.bool, count=1)[0]
		mean[i] = np.fromfile(f, dtype=np.float64, count=N_dim)
		cov[i] = np.fromfile(f, dtype=np.float64, count=N_dim*N_dim)
		tmp = np.fromfile(f, dtype=np.float64, count=N_dim*(N_dim+1))
		tmp = np.fromfile(f, dtype=np.uint64, count=1)
	
	cov.shape = (N_files, N_dim, N_dim)
	
	return converged, mean, cov


# Load binned probability density functions (pdfs) from a galstar bin output file
def load_bins(fname):
	f = open(fname, 'rb')
	
	# Read in header
	N_files = np.fromfile(f, dtype=np.uint32, count=1)
	bin_width = np.fromfile(f, dtype=np.uint32, count=2)
	bin_min = np.fromfile(f, dtype=np.float64, count=2)
	bin_max = np.fromfile(f, dtype=np.float64, count=2)
	bin_dx = np.fromfile(f, dtype=np.float64, count=2)
	
	# Read in pdfs
	bin_data = np.fromfile(f, dtype=np.float64)#, count=N_files*bin_width[0]*bin_width[1])
	N_files_empirical = bin_data.size / bin_width[0] / bin_width[1]
	bin_data.shape = (N_files_empirical, bin_width[0], bin_width[1])
	#print N_files, N_files_empirical
	
	f.close()
	
	# Create list containing bounds
	bounds = [bin_min[0], bin_max[0], bin_min[1], bin_max[1]]
	
	return bounds, bin_data


# Same as above, but for a gzipped file
def load_bins_gzip(fname):
	f_tmp = gzip.open(fname, 'rb')
	f = f_tmp.read()
	f_tmp.close()
	
	# Read in header
	N_files = np.fromstring(f[0:4], dtype=np.uint32, count=1)
	bin_width = np.fromstring(f[4:12], dtype=np.uint32, count=2)
	bin_min = np.fromstring(f[12:28], dtype=np.float64, count=2)
	bin_max = np.fromstring(f[28:44], dtype=np.float64, count=2)
	bin_dx = np.fromstring(f[44:60], dtype=np.float64, count=2)
	
	# Read in pdfs
	bin_data = np.fromstring(f[60:], dtype=np.float64)
	N_files_empirical = bin_data.size / bin_width[0] / bin_width[1]
	bin_data.shape = (N_files_empirical, bin_width[0], bin_width[1])
	#print N_files, N_files_empirical
	
	# Create list containing bounds
	bounds = [bin_min[0], bin_max[0], bin_min[1], bin_max[1]]
	
	return bounds, bin_data


# Smooth binned data with Gaussian kernel
def smooth_bins(p, sigma):
	# Apply Gaussian smoothing to each image
	p_smooth = np.empty(p.shape, dtype=np.float64)
	filters.gaussian_filter(p, np.insert(sigma, 0, 0), output=p_smooth, mode='nearest')
	
	# Normalize each image to unit probability
	for i in xrange(p_smooth.shape[0]):
		p_smooth[i] /= np.sum(p_smooth[i])
	
	return p_smooth




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
	width = float(pdfs.shape[1])
	guess = np.log(np.random.ranf(N_regions) * 2.*width/float(N_regions))	# Zeroeth-order guess
	guess = opt.fmin(chi_leastsq, guess, args=(pdf_stacked, chimax, regulator), ftol=1.e-6, maxiter=100000, maxfun=1e8)	# A better guess
	print 'guess:', np.exp(guess)
	chi = chi_leastsq(guess, pdfs, chimax=5.)
	print 'chi^2:', np.sum(chi*chi)
	
	# Do the full fit
	x, success = opt.leastsq(chi_leastsq, guess, args=(pdfs, chimax, regulator), ftol=1.e-6, maxfev=10000)
	measure = chi_leastsq(x, pdfs, chimax, regulator)
	
	return x, success, guess, measure


# Return a measure to minimize by simulated annealing
def anneal_measure(log_Delta_y, pdfs, p0=1.e-3, regulator=10000.):
	Delta_y = np.exp(log_Delta_y)
	
	measure = line_integral(Delta_y, pdfs)				# Begin with line integral through each stellar pdf
	measure = p0 * np.log(2. * np.cosh(measure / p0))	# Soften around zero (measure -> positive const. below scale p0)
	measure = -np.sum(np.log(measure))					# Sum logarithms of line integrals
	
	# Disfavor larger values of Delta_y slightly
	#measure += np.sum(Delta_y*Delta_y) / (regulator*regulator)
	
	# Disfavor larger values of ln(Delta_y) slightly
	measure += np.sum(log_Delta_y*log_Delta_y) / (2.*regulator*regulator)
	
	#print measure
	return measure


# Maximize the line integral by simulated annealing
def min_anneal(pdfs, N_regions=15, p0=1.e-3, regulator=10000.):
	# Start with random guess
	width = float(pdfs.shape[1])
	guess = np.log(np.random.ranf(N_regions) * 2.* width/float(N_regions))
	
	# Set bounds on step size in Delta_Ar
	lower = np.empty(N_regions, dtype=np.float64)
	upper = np.empty(N_regions, dtype=np.float64)
	lower.fill(-0.01)
	upper.fill(0.01)
	
	# Run simulated annealing
	x, success = opt.anneal(anneal_measure, guess, args=(pdfs, p0, regulator), lower=lower, upper=upper, feps=1.e-12, maxiter=1000, dwell=200)
	measure = anneal_measure(x, pdfs, p0, regulator)
	
	return x, success, guess, measure


# Fit line-of-sight reddening profile, given the binned pdfs in <bin_fname> and stats in <stats_fname>
def fit_los(bin_fname, stats_fname, N_regions, converged=False, method='anneal', smooth=1, regulator=10000.):
	# Load pdfs
	print 'Loading binned pdfs...'
	bounds, p = None, None
	if '.gzip' in bin_fname:
		bounds, p = load_bins_gzip(abspath(bin_fname))
	else:
		bounds, p = load_bins(abspath(bin_fname))
	if converged:
		converged, means, cov = load_stats(abspath(stats_fname))
		p = smooth_bins(p[converged], [smooth,smooth])
	else:
		p = smooth_bins(p, [smooth,smooth])
	
	# Fit reddening profile
	x, succes, guess, measure = None, None, None, None
	if method == 'leastsq':
		print 'Fitting reddening profile using the LM method (scipy.optimize.leastsq)...'
		x, success, guess, measure = min_leastsq(p, N_regions=N_regions, chimax=5., regulator=regulator)
	elif method == 'anneal':
		print 'Fitting reddening profile using simulated annealing (scipy.optimize.anneal)...'
		x, success, guess, measure = min_anneal(p, N_regions=N_regions, p0=1.e-3, regulator=regulator)
	
	# Convert output into physical coordinates (rather than pixel coordinates)
	Delta_Ar = np.exp(x) * ((bounds[3] - bounds[2]) / float(p.shape[1]))
	guess = np.exp(guess) * ((bounds[3] - bounds[2]) / float(p.shape[1]))
	
	# Output basic information about fit
	print 'Delta_Ar:', Delta_Ar
	print 'success:', success
	print 'measure:', measure
	
	return bounds, p, measure, success, Delta_Ar, guess




#
# PLOTS
#

# Overplot reddening profile on stacked pdfs
def plot_profile(bounds, p, Delta_Ar, plot_fn=None):
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
	img = np.average(p, axis=0).T
	img /= np.max(img, axis=0)
	img.shape = (1, p.shape[1], p.shape[2])
	ax.imshow(img[0], extent=bounds, origin='lower', aspect='auto', cmap='hot')
	ax.plot(mu_anchors, Ar_anchors)
	ax.set_xlim(bounds[0], bounds[1])
	ax.set_ylim(bounds[2], bounds[3])	
	ax.set_xlabel(r'$\mu$', fontsize=18)
	ax.set_ylabel(r'$A_r$', fontsize=18)
	fig.subplots_adjust(bottom=0.10)
	
	if plot_fn != None:
		fig.savefig(abspath(plot_fn), dpi=150)


# Save the reddening profile to an ASCII file
def save_profile(fname, bounds, Delta_Ar):
	# Calculate reddening profile
	N_regions = Delta_Ar.size
	mu_anchors = np.linspace(bounds[0], bounds[1], N_regions+1)
	Ar_anchors = np.empty(N_regions+1, dtype=Delta_Ar.dtype)
	for i in xrange(N_regions+1):
		Ar_anchors[i] = bounds[2] + np.sum(Delta_Ar[:i])
	
	output = np.empty((2, N_regions+1), dtype=np.float64)
	output[0] = mu_anchors
	output[1] = Ar_anchors
	
	# Write to file
	np.savetxt(abspath(fname), output)




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
	parser.add_argument('-o', '--outfn', type=str, default=None, help='Output filename for reddening profile.')
	parser.add_argument('-po', '--plotfn', type=str, default=None, help='Filename for plot of result.')
	parser.add_argument('-sh', '--show', action='store_true', help='Show plot of result.')
	parser.add_argument('-sm', '--smooth', type=int, default=1, help='Std. dev. of smoothing kernel (in pixels) for individual pdfs (default: 1).')
	parser.add_argument('-reg', '--regulator', type=float, default=10000., help='Width of support of prior on ln(Delta_Ar) (default: 10000).')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	np.seterr(all='ignore')
	
	tstart = time()
	bounds, p, measure, success, Delta_Ar, guess = fit_los(values.binfn, values.statsfn, values.N, converged=values.converged, method=values.method, smooth=values.smooth, regulator=values.regulator)
	duration = time() - tstart
	print 'Time elapsed: %.1f s' % duration
	
	if values.outfn != None:
		save_profile(values.outfn, bounds, Delta_Ar)
	
	if (values.plotfn != None) or (values.show != None):
		plot_profile(bounds, p, Delta_Ar, values.plotfn)
	
	if values.show:
		plt.show()
	
	# Show the guess
	#plot_profile(bounds, p, guess)
	#plt.show()
	
	return 0


if __name__ == '__main__':
	main()

