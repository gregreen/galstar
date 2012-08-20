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

# TODO: Add option to tie pixel to adjacent pixels in input map

import sys, argparse
from os.path import abspath, exists
from time import time

import numpy as np
import scipy.ndimage.filters as filters
from scipy import weave
import scipy.optimize

import healpy as hp

import nlopt

import matplotlib as mplib
import matplotlib.pyplot as plt

from galstar_io import *
from galstarutils import get_objects
import healpix_utils as hputils



#
# OPTIMIZATION ROUTINES
#

# Compute the line integral through multiple images, stacked in <img>
def line_integral(Delta_y, img):
	# Determine the number of bins per piecewise linear region
	N_regions = Delta_y.shape[0] - 1
	if img.shape[1] % N_regions != 0:
		raise Exception('Number of samples in mu (%d) not integer multiple of number of piecewise linear regions (%d).' % (img.shape[1], (Delta_y.shape[0] - 1)))
	N_images = img.shape[0]
	y_max = img.shape[2]
	N_samples = img.shape[1] / N_regions
	
	line_int_ret = np.zeros(N_images, dtype=np.float64)
	code = """
		//double tmp = Delta_y(0);
		double y = Delta_y(0);
		double y_ceil, y_floor, dy;
		int x = 0;
		
		//if((int)verbose != -1) {
		//	std::cout << "Delta_y = (";
		//	for(int i=0; i<N_regions+1; i++) {
		//		std::cout << 20./1000.*Delta_y(i) << (i < N_regions ? ", " : "");
		//	}
		//	std::cout << std::endl;
		//}
		
		for(int i=0; i<N_regions; i++) {
			dy = (double)(Delta_y(i+1)) / (double)(N_samples);
			//std::cout << "(" << x << ", " << y << ", " << tmp << ") ";
			for(int j=0; j<N_samples; j++, x++, y+=dy) {
				y_ceil = ceil(y);
				y_floor = floor(y);
				if((int)y_ceil >= y_max) { break; }
				if((int)y_floor < 0) { break; }
				for(int k=0; k<N_images; k++) {
					line_int_ret(k) += (y_ceil - y) * img(k, x, (int)y_floor) + (y - y_floor) * img(k, x, (int)y_ceil);
					//if((int)verbose == k) {
					//	std::cout << "p(" << 5. + 15./120.*x << ", " << 20./1000.*y << ") = " << (y_ceil - y) * img(k, x, (int)y_floor) + (y - y_floor) * img(k, x, (int)y_ceil) << std::endl;
					//}
				}
			}
			//tmp += (double)(Delta_y(i+1));
			if((int)y_ceil >= y_max) { break; }
			if((int)y_floor < 0) { break; }
		}
		//std::cout << "(" << x << ", " << y << ", " << tmp << ") ";
		//std::cout << std::endl;
		return_val = x;
	"""
	x = weave.inline(code, ['img', 'Delta_y', 'N_images', 'N_regions', 'N_samples', 'y_max', 'line_int_ret'], type_converters=weave.converters.blitz, compiler='gcc')
	
	#print np.sum(Delta_y)
	#if verbose != -1:
	#	print ''
	
	#if np.random.random() < 0.001:
	#	print x
	
	#if x > 120:
	#	print x
	
	return line_int_ret


# Return chi for the model with steps in reddening given by <log_Delta_y>
def chi_leastsq(log_Delta_y, pdfs=None, p0=1.e-5, regulator=10000.):
	Delta_y = np.exp(log_Delta_y)
	
	measure = line_integral(Delta_y, pdfs)	# Begin with line integral through each stellar pdf
	measure += p0 * np.exp(-measure/p0)		# Soften around zero (measure -> p0 const. below scale p0)
	measure = -2. * np.log(measure)
	
	# Disfavor larger values of ln(Delta_y) slightly
	#bias = 0.
	#measure += np.sum((log_Delta_y[1:]-bias)*(log_Delta_y[1:]-bias)) / (2.*regulator*regulator)
	measure += np.sum(Delta_y[1:]*Delta_y[1:]) / (2.*regulator*regulator)
	
	return np.sqrt(measure)


# Minimize chi^2 for a line running through the given pdfs
def min_leastsq(pdfs, guess, p0=1.e-5, regulator=10000.):
	N_regions = guess.size - 1
	
	sys.stderr.write('Guess: %s\n' % np.array_str(guess, max_line_width=N_regions*100, precision=8))
	chi = chi_leastsq(np.log(guess), pdfs, p0=p0)
	print 'chi^2 of guess:', np.sum(chi*chi)
	
	# Do the full fit
	x, success = scipy.optimize.leastsq(chi_leastsq, np.log(guess), args=(pdfs, p0, regulator), ftol=1.e-6, maxfev=10000)
	measure = chi_leastsq(x, pdfs, p0, regulator)
	
	return np.exp(x), success, np.sum(measure)


# Return a measure to minimize by simulated annealing
def anneal_measure(log_Delta_y, pdfs, p0=1.e-5, regulator=1000.):
	Delta_y = np.exp(log_Delta_y)
	if np.any(np.isnan(Delta_y)):
		raise ValueError('Delta_y contains NaN values.')
	
	measure = line_integral(Delta_y, pdfs)	# Begin with line integral through each stellar pdf
	measure += p0 * np.exp(-measure/p0)		# Soften around zero (measure -> positive const. below scale p0)
	measure = -np.sum(np.log(measure))		# Sum logarithms of line integrals
	
	# Disfavor larger values of ln(Delta_y) slightly
	#bias = 0.
	#measure += np.sum((log_Delta_y[1:]-bias)*(log_Delta_y[1:]-bias)) / (2.*regulator*regulator)
	measure += np.sum(Delta_y[1:]*Delta_y[1:]) / (2.*regulator*regulator)
	
	return measure


# Maximize the line integral by simulated annealing
def min_anneal(pdfs, guess, p0=1.e-5, regulator=1000., dwell=1000):
	N_regions = guess.size - 1
	
	# Set bounds on step size in Delta_Ar
	lower = np.empty(N_regions+1, dtype=np.float64)
	upper = np.empty(N_regions+1, dtype=np.float64)
	lower.fill(-0.01)
	upper.fill(0.01)
	
	# Run simulated annealing
	#feps=1.e-12
	x, success = scipy.optimize.anneal(anneal_measure, np.log(guess), args=(pdfs, p0, regulator), lower=lower, upper=upper, maxiter=1000, dwell=dwell)
	measure = anneal_measure(x, pdfs, p0, regulator)
	
	return np.exp(x), success, measure


# Return a measure to minimize with NLopt
def nlopt_measure(Delta_y, grad, pdfs, p0=1.e-5, regulator=1000., Delta_y_neighbor=None, weight_neighbor=None):
	if grad.size > 0:
		raise Exception('Gradient-free methods only, please!')
	
	#Delta_y = np.exp(log_Delta_y)
	
	measure = line_integral(Delta_y, pdfs)	# Begin with line integral through each stellar pdf
	measure += p0 * np.exp(-measure/p0)		# Soften around zero (measure -> positive const. below scale p0)
	measure = -np.sum(np.log(measure))		# Sum logarithms of line integrals
	
	# Disfavor larger values of ln(Delta_y) slightly
	#bias = 0.
	#log_Delta_y = np.log(Delta_y)
	#measure += np.sum((log_Delta_y[1:]-bias)*(log_Delta_y[1:]-bias)) / (2.*regulator*regulator)
	#print np.sum((log_Delta_y[1:]-bias)*(log_Delta_y[1:]-bias)) / (2.*regulator*regulator)
	#print log_Delta_y[1:]
	#print ''
	measure += np.sum(Delta_y[1:]*Delta_y[1:]) / (2.*regulator*regulator)
	
	# Tie this pixel to neighbors
	if Delta_y_neighbor != None:
		Delta_y_tension = weight_neighbor * (Delta_y_neighbor - Delta_y).T / (2. * 10. * 10.)
		measure += np.sum(Delta_y_tension * Delta_y_tension)
	
	return measure


# Maximize the line integral using an algorithm from NLopt
def min_nlopt(pdfs, guess, p0=1.e-5, regulator=1000., maxtime=25., maxeval=10000, algorithm='CRS', Delta_Ar_neighbor=None, weight_neighbor=None):
	N_regions = guess.size - 1
	
	opt = None
	if algorithm == 'CRS':
		opt = nlopt.opt(nlopt.GN_CRS2_LM, N_regions+1)
	elif algorithm == 'MLSL':
		opt = nlopt.opt(nlopt.G_MLSL_LDS, N_regions+1)
	
	# Set lower and upper bounds on Delta_Ar
	lower = np.empty(N_regions+1, dtype=np.float64)
	upper = np.empty(N_regions+1, dtype=np.float64)
	lower.fill(1.e-10)
	upper.fill(max(float(pdfs.shape[2]), 1.2*np.max(guess)))
	opt.set_lower_bounds(lower)
	opt.set_upper_bounds(upper)
	
	# Set local optimizer (if required)
	if algorithm == 'MLSL':
		local_opt = nlopt.opt(nlopt.LN_COBYLA, N_regions+1)
		local_opt.set_lower_bounds(lower)
		local_opt.set_upper_bounds(upper)
		local_opt.set_initial_step(15.)
		opt.set_local_optimizer(local_opt)
	
	opt.set_initial_step(15.)
	
	# Set stopping conditions
	opt.set_maxtime(maxtime)
	opt.set_maxeval(maxeval)
	#opt.set_xtol_abs(0.1)
	
	# Set the objective function
	opt.set_min_objective(lambda x, grad: nlopt_measure(x, grad, pdfs, p0, regulator, Delta_Ar_neighbor, weight_neighbor))
	
	# Run optimization algorithm
	x = opt.optimize(guess)
	measure = opt.last_optimum_value()
	success = opt.last_optimize_result()
	
	return x, success, measure


def min_brute(pdfs, guess, p0=1.e-5, regulator=10000.):
	N_regions = guess.size - 1
	ranges = [(-5., 5.) for i in xrange(N_regions+1)]
	
	x = scipy.optimize.brute(anneal_measure, ranges, args=(pdfs, p0, regulator), Ns=5)
	measure = anneal_measure(x, pdfs, p0, regulator)
	
	return np.exp(x), 0, measure


def guess_measure(Delta_y, y_mean, y_err, weight):
	y_profile = np.empty(Delta_y.size, np.float64)
	y_profile[0] = Delta_y[0]
	for i in xrange(1, Delta_y.size):
		y_profile[i] = y_profile[i-1] + Delta_y[i]
	y_profile -= y_mean
	y_profile = np.divide(y_profile, y_err)
	return np.sum(weight * y_profile * y_profile)


def gen_guess(pdfs, N_regions=15):
	pdfs_flat = np.sum(pdfs, axis=0)
	
	y_weight = np.empty(pdfs_flat.shape, dtype=np.float64)
	y2_weight = np.empty(pdfs_flat.shape, dtype=np.float64)
	for i in xrange(pdfs_flat.shape[1]):
		y_weight[:,i] = i * pdfs_flat[:,i]
		y2_weight[:,i] = i*i * pdfs_flat[:,i]
	mu_y = np.divide(np.sum(y_weight, axis=1), np.sum(pdfs_flat, axis=1))
	mu_y2 = np.divide(np.sum(y2_weight, axis=1), np.sum(pdfs_flat, axis=1))
	mu_y[np.isnan(mu_y)] = 0.
	mu_y2[np.isnan(mu_y2)] = 0.
	
	y_mean = np.empty(N_regions+1, dtype=np.float64)
	y_err = np.empty(N_regions+1, dtype=np.float64)
	weight = np.empty(N_regions+1, dtype=np.float64)
	
	for i in xrange(N_regions+1):
		x_0 = int((float(i) - 0.5) * float(pdfs.shape[1]) / float(N_regions))
		x_1 = int((float(i) + 0.5) * float(pdfs.shape[1]) / float(N_regions))
		if x_0 < 0:
			x_0 = 0
		if x_1 >= pdfs.shape[1]:
			x_1 = pdfs.shape[1]
		
		weight[i] = np.sum(pdfs_flat[x_0:x_1,:])
		y_mean[i] = np.mean(mu_y[x_0:x_1])
		y2_mean = np.mean(mu_y2[x_0:x_1])
		y_err[i] = np.sqrt(y2_mean - y_mean[i]*y_mean[i])
	
	y_mean[~np.isfinite(y_mean)] = 0.
	y_err[y_mean < 1.e-5] = np.inf
	
	opt = nlopt.opt(nlopt.G_MLSL_LDS, N_regions+1)
	
	# Set lower and upper bounds on Delta_Ar
	lower = np.empty(N_regions+1, dtype=np.float64)
	upper = np.empty(N_regions+1, dtype=np.float64)
	lower.fill(0.)
	upper.fill(float(pdfs.shape[2]))
	opt.set_lower_bounds(lower)
	opt.set_upper_bounds(upper)
	
	local_opt = nlopt.opt(nlopt.LN_COBYLA, N_regions+1)
	local_opt.set_lower_bounds(lower)
	local_opt.set_upper_bounds(upper)
	opt.set_local_optimizer(local_opt)
	
	# Set stopping conditions
	opt.set_maxtime(2.)
	
	# Set the objective function
	opt.set_min_objective(lambda x, grad: guess_measure(x, y_mean, y_err, weight))
	
	# Start with random guess
	guess = 3.0 * (np.random.ranf(N_regions+1) * np.max(y_mean)/float(N_regions+1)).astype(np.float64)
	guess[np.isinf(y_err)] = 0.01
	
	# Run optimization algorithm
	x = opt.optimize(guess)
	measure = opt.last_optimum_value()
	success = opt.last_optimize_result()
	
	x[(x < 1.e-5)] = 1.e-5
	Delta_y_mean = np.empty(N_regions+1, dtype=np.float64)
	Delta_y_mean[0] = y_mean[0]
	Delta_y_mean[1:] = y_mean[1:] - y_mean[:-1]
	
	return x, Delta_y_mean


# Fit line-of-sight reddening profile, given the binned pdfs in <bin_fname> and stats in <stats_fname>
def fit_los(bin_fname, stats_fname, N_regions, sparse=True, converged=False, method='anneal', smooth=(1,1), regulator=10000., dwell=1000, maxtime=25., maxeval=10000, p0=1.e-5, ev_range=25., iterate=None):
	# Load pdfs
	sys.stderr.write('Loading binned pdfs...\n')
	bounds, p = load_bins(bin_fname, sparse)
	mask = np.logical_not(np.sum(np.sum(np.logical_not(np.isfinite(p)), axis=1), axis=1).astype(np.bool))	# Filter out images with NaN bins
	converged_arr, ln_evidence, means, cov = load_stats(stats_fname)
	ln_evidence_cutoff = np.max(ln_evidence) - ev_range
	mask = np.logical_and(mask, (ln_evidence > ln_evidence_cutoff))	# Filter out objects which do not appear to fit the stellar model
	if converged:	# Filter out nonconverged images
		mask = np.logical_and(mask, converged_arr)			# Filter out stars which did not converge
	sys.stderr.write('# of stars filtered out: %d of %d.\n\n' % (np.sum(~mask), p.shape[0]))
	p = smooth_bins(p[mask], smooth)
	
	# Load in neighboring pixels from previous iteration
	Delta_Ar_neighbor, weight_neighbor = None, None
	if iterate != None:
		Delta_Ar_neighbor, weight_neighbor = get_neighbors(abspath(iterate[0]), int(iterate[1]))
		Delta_Ar_neighbor *= float(p.shape[2]) / (bounds[3] - bounds[2])
	
	# Generate a guess based on the stacked pdfs
	sys.stderr.write('Generating guess...\n')
	guess, y_mean = gen_guess(p, N_regions=N_regions)
	guess_Delta_Ar = guess * ((bounds[3] - bounds[2]) / float(p.shape[2]))
	Delta_Ar_mean = y_mean * ((bounds[3] - bounds[2]) / float(p.shape[2]))
	guess_fitness = nlopt_measure(guess, np.array([]), p, p0, regulator, Delta_Ar_neighbor, weight_neighbor)
	sys.stderr.write('Guess: %s\n' % np.array_str(guess_Delta_Ar, max_line_width=N_regions*100, precision=8))
	sys.stderr.write('Guess measure: %.3f\n\n' % guess_fitness)
	guess_line_int = line_integral(guess, p)
	
	# Fit reddening profile
	x, success, measure = None, None, None
	if method == 'leastsq':
		sys.stderr.write('Fitting reddening profile using the LM method (scipy.optimize.leastsq)...\n')
		x, success, measure = min_leastsq(p, guess, p0=p0, regulator=regulator)
	elif method == 'anneal':
		sys.stderr.write('Fitting reddening profile using simulated annealing (scipy.optimize.anneal)...\n')
		x, success, measure = min_anneal(p, guess, p0=p0, regulator=regulator, dwell=dwell)
	elif method == 'brute':
		sys.stderr.write('Fitting reddening profile by brute force (scipy.optimize.brute)...\n')
		x, success, measure = min_brute(p, guess, p0=p0, regulator=regulator)
	elif method == 'nlopt MLSL':
		sys.stderr.write('Fitting reddening profile using NLopt (nlopt.G_MLSL_LDS with local optimizer nlopt.LN_COBYLA)...\n')
		x, success, measure = min_nlopt(p, guess, p0=p0, regulator=regulator, maxtime=maxtime, maxeval=maxeval, algorithm='MLSL')
	elif method == 'nlopt CRS':
		sys.stderr.write('Fitting reddening profile using NLopt (nlopt.GN_CRS2_LM)...\n')
		x, success, measure = min_nlopt(p, guess, p0=p0, regulator=regulator, maxtime=maxtime, maxeval=maxeval, algorithm='CRS')
	
	measure = nlopt_measure(x, np.array([]), p, p0, regulator, Delta_Ar_neighbor, weight_neighbor)
	line_int = line_integral(x, p)
	N_outliers = np.sum(line_int == 0.)
	N_softened = np.sum(line_int < p0)
	
	# Convert output into physical coordinates (rather than pixel coordinates)
	Delta_Ar = x * ((bounds[3] - bounds[2]) / float(p.shape[2]))
	
	# Output basic information about fit
	sys.stderr.write('Delta_Ar: %s\n' % np.array_str(Delta_Ar, max_line_width=N_regions*100, precision=8))
	sys.stderr.write('success: %d\n' % success)
	sys.stderr.write('measure: %f\n' % measure)
	sys.stderr.write('Extreme outliers: %d of %d\n' % (N_outliers, line_int.size))
	sys.stderr.write('Outliers (below softening limit): %d of %d\n\n' % (N_softened, line_int.size))
	
	return bounds, p, line_int, guess_line_int, measure, success, Delta_Ar, guess_Delta_Ar, Delta_Ar_mean




#
# PLOTS
#

# Overplot reddening profile on stacked pdfs
def plot_profile(bounds, p, Delta_Ar, plot_fn=None, overplot=None):
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
	
	# Stack pdfs
	img = np.average(p, axis=0)
	
	# Determine maximum reddening present in pdfs
	y_index_max = np.max(np.where(img > 0)[1])
	max_Ar = y_index_max.astype(np.float64) / float(p.shape[2]) * (bounds[3] - bounds[2]) + bounds[2]
	
	# Plot stacked pdfs
	img = img.T
	img /= np.max(img, axis=0)
	#p0 = np.mean(img[img > 0])
	#img += p0 * np.exp(-img / p0)
	#img = np.log(img)
	#img -= np.max(img, axis=0)
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
	if type(Delta_Ar) is not list:
		Delta_Ar = [Delta_Ar]
	for n in xrange(len(Delta_Ar)):
		N_regions = Delta_Ar[n].size - 1
		mu_anchors = np.linspace(bounds[0], bounds[1], N_regions+1)
		Ar_anchors = np.empty(N_regions+1, dtype=np.float64)
		Ar_anchors[0] = bounds[2] + Delta_Ar[n][0]
		for i in xrange(1, N_regions+1):
			Ar_anchors[i] = Ar_anchors[i-1] + Delta_Ar[n][i]
		ax.plot(mu_anchors, Ar_anchors)
	
	# Set axis limits and labels
	y_max = min([bounds[3], max_Ar])
	ax.set_xlim(bounds[0], bounds[1])
	ax.set_ylim(bounds[2], y_max)
	ax.set_xlabel(r'$\mu$', fontsize=18)
	ax.set_ylabel(r'$A_r$', fontsize=18)
	fig.subplots_adjust(bottom=0.10)
	
	if plot_fn != None:
		fig.savefig(abspath(plot_fn), dpi=150)


# Overplot reddening profile on stacked pdfs
def plot_indiv(bounds, p, Delta_Ar, line_int):
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# Determine the line-of-sight reddening profile
	if type(Delta_Ar) is not list:
		Delta_Ar = [Delta_Ar]
	if type(line_int) is not list:
		line_int = [line_int]
	
	mu_anchors = []
	Ar_anchors = []
	
	for n, DAr in enumerate(Delta_Ar):
		N_regions = Delta_Ar[n].size - 1
		mu_anchors.append(np.linspace(bounds[0], bounds[1], N_regions+1))
		Ar_anchors.append(np.empty(N_regions+1, dtype=np.float64))
		Ar_anchors[-1][0] = bounds[2] + DAr[0]
		for i in xrange(1, N_regions+1):
			Ar_anchors[-1][i] = Ar_anchors[-1][i-1] + DAr[i]
	
	# Determine maximum reddening present in pdfs
	img = np.average(p, axis=0)
	y_index_max = np.max(np.where(img > 0)[1])
	max_Ar = y_index_max.astype(np.float64) / float(p.shape[2]) * (bounds[3] - bounds[2]) + bounds[2]
	y_max = min([bounds[3], max_Ar])
	
	# Plot pdf and reddening profile for each star
	fig = None
	x_text_left = bounds[0] + 0.05 * (bounds[1] - bounds[0])
	x_text_right = bounds[0] + 0.9 * (bounds[1] - bounds[0])
	y_text = bounds[2] + 0.95 * (y_max - bounds[2])
	palette = ['b', 'g', 'r', 'c', 'y']
	for i in xrange(p.shape[0]):
		ax_index = (i % 6) + 1
		if ax_index == 1:
			fig = plt.figure(figsize=(8.5,11), dpi=100)
		ax = fig.add_subplot(3, 2, ax_index)
		ax.imshow(p[i].T, extent=bounds, origin='lower', aspect='auto', cmap='hot')
		
		for k, (mu, Ar) in enumerate(zip(mu_anchors, Ar_anchors)):
			ax.plot(mu, Ar, palette[k])
		
		ax.text(x_text_left, y_text, r'$%d$' % i, color='w', fontsize=14, horizontalalignment='left', verticalalignment='top')
		
		for k, l_int in enumerate(line_int):
			ax.text(x_text_right, y_text - k * 0.1 * (y_max - bounds[2]), r'$%.2g$' % l_int[i], color=palette[k], fontsize=14, horizontalalignment='right', verticalalignment='top')
		
		# Set axis limits and labels
		ax.set_xlim(bounds[0], bounds[1])
		ax.set_ylim(bounds[2], y_max)
		ax.set_xlabel(r'$\mu$', fontsize=18)
		ax.set_ylabel(r'$A_r$', fontsize=18)
	
	fig.show()


def output_profile(fname, pixnum, bounds, Delta_Ar, N_stars, line_int, measure, success):
	'''
	Append the reddening profile to the end of the binary file given by <fname>.
	
	Format - for each pixel:
		pixnum		(uint64)
		N_stars		(uint32)
		measure		(float64)
		success		(uint16)
		N_regions	(uint16)
		line_int	(float64) x N_stars
		mu_anchors	(float64) x (N_regions + 1)
		Ar_anchors	(float64) x (N_regions + 1)
	'''
	
	# Calculate reddening profile
	N_regions = Delta_Ar.size - 1
	mu_anchors = np.linspace(bounds[0], bounds[1], N_regions+1).astype(np.float64)
	Ar_anchors = np.empty(N_regions+1, dtype=np.float64)
	Ar_anchors[0] = bounds[2] + Delta_Ar[0]
	for i in xrange(1, N_regions+1):
		Ar_anchors[i] = bounds[2] + np.sum(Delta_Ar[:i])
	
	# Append to end of file <fname>
	f = open(fname, 'ab')
	f.write(np.array([pixnum], dtype=np.uint64).tostring())
	f.write(np.array([N_stars], dtype=np.uint32).tostring())
	f.write(np.array([measure], dtype=np.float64).tostring())
	f.write(np.array([success, N_regions], dtype=np.uint16).tostring())
	f.write(line_int.tostring())
	f.write(mu_anchors.tostring())
	f.write(Ar_anchors.tostring())
	f.close()



#
# Load in neighboring pixels
#

def get_neighbors(map_fname, pixindex, mu_anchors=None):
	m = hputils.ExtinctionMap(map_fname, FITS=True)
	
	if mu_anchors == None:
		mu_anchors = m.mu
	
	# Query neighboring pixels
	neighbor_index = hp.pixelfunc.get_all_neighbours(m.nside, pixindex, nest=m.nested)
	Delta_Ar = m.evaluate(mu_anchors, pix_index=neighbor_index)
	Delta_Ar[1:] = Delta_Ar[1:] - Delta_Ar[:-1]
	mask = np.isfinite(Delta_Ar[0,:])
	neighbor_index = neighbor_index[mask]
	Delta_Ar = Delta_Ar[:,mask]
	
	# Assign weight to each pixel based on distance
	theta, phi = hp.pix2ang(m.nside, neighbor_index, nest=m.nested)
	theta_0, phi_0 = hp.pix2ang(m.nside, pixindex, nest=m.nested)
	print theta - theta_0
	print phi - phi_0
	dist = np.arccos(np.sin(theta_0) * np.sin(theta) + np.cos(theta_0) * np.cos(theta) * np.cos(phi - phi_0))
	sigma_dist = hp.pixelfunc.nside2resol(m.nside, arcmin=False)
	weight = np.exp(-dist * dist / (2. * sigma_dist * sigma_dist))
	weight /= np.sum(weight)
	
	# Return reddening in each bin for each neighboring pixel, as well as weight assigned to each neighbor
	return Delta_Ar.T, weight




#
# MAIN
#

def main():
	parser = argparse.ArgumentParser(prog='fit_pdfs.py', description='Fit line-of-sight reddening law from probability density functions of individual stars.', add_help=True)
	parser.add_argument('binfn', type=str, help='File containing binned probability density functions for each star along l.o.s. (also accepts gzipped files)')
	parser.add_argument('statsfn', type=str, help='File containing summary statistics for each star.')
	parser.add_argument('-N', '--N', type=int, default=20, help='# of piecewise-linear regions in DM-Ar relation (default: 20)')
	parser.add_argument('-mtd', '--method', type=str, choices=('anneal', 'leastsq', 'brute', 'nlopt CRS', 'nlopt MLSL'), default='nlopt CRS', help='Optimization method (default: nlopt CRS)')
	parser.add_argument('-cnv', '--converged', action='store_true', help='Filter out unconverged stars.')
	parser.add_argument('-sm', '--smooth', type=float, nargs=2, default=(2,2), help='Std. dev. of smoothing kernel (in pixels) for individual pdfs (default: 2 2).')
	parser.add_argument('-reg', '--regulator', type=float, default=1000., help='Width of support of prior on Delta_Ar (default: 1000).')
	parser.add_argument('-o', '--outfn', type=str, nargs=2, default=None, help='Output filename for reddening profile and healpix pixel number.')
	parser.add_argument('-po', '--plotfn', type=str, default=None, help='Filename for plot of result.')
	parser.add_argument('-sh', '--show', action='store_true', help='Show plot of result.')
	parser.add_argument('-ovp', '--overplot', type=str, default=None, help='Overplot true values from galfast FITS file')
	parser.add_argument('-dw', '--dwell', type=int, default=1000, help='dwell parameter for annealing algorithm. The higher the value, the greater the chance of convergence (default: 1000).')
	parser.add_argument('-W', '--maxtime', type=float, default=100., help='Maximum walltime (in seconds) for NLopt routines (default: 100).')
	parser.add_argument('-M', '--maxeval', type=int, default=10000, help='Maximum # of evaluations for NLopt routines (default: 10000).')
	parser.add_argument('-p0', '--floor', type=float, default=5.e-3, help='Floor on stellar line integrals (default: 5.e-3).')
	parser.add_argument('-ev', '--evidence_range', type=float, default=25., help='Maximum difference in ln(evidence) from max. value before star is considered outlier (default: 25).')
	parser.add_argument('-nsp', '--nonsparse', action='store_true', help='Binned pdfs are not stored in sparse format.')
	parser.add_argument('-pltind', '--plot_individual', type=int, nargs=2, default=None, help='Plot individual pdfs with reddening profile.')
	parser.add_argument('-it', '--iterate', type=str, nargs=2, default=None, help='Tie pixel to neighbors in given reddening map. The healpix index of this pixel must be provided as the second argument.')
	#parser.add_argument('-v', '--verbose', action='store_true', help='Print information on fit.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	np.seterr(all='ignore')
	
	tstart = time()
	
	# Fit the line of sight
	bounds, p, line_int, guess_line_int, measure, success, Delta_Ar, guess, Delta_Ar_mean = fit_los(values.binfn, values.statsfn, values.N, sparse=(not values.nonsparse), converged=values.converged, method=values.method, smooth=values.smooth, regulator=values.regulator, dwell=values.dwell, maxtime=values.maxtime, maxeval=values.maxeval, p0=values.floor, ev_range=values.evidence_range, iterate=values.iterate)
	duration = time() - tstart
	sys.stderr.write('Time elapsed: %.1f s\n' % duration)
	
	# Save the reddening profile to an ASCII file, or print to stdout
	N_stars = p.shape[0]
	if values.outfn != None:
		output_profile(values.outfn[0], int(values.outfn[1]), bounds, Delta_Ar, N_stars, line_int, measure, success)
	
	# Plot individual reddening profile
	if values.plot_individual != None:
		i1, i2 = values.plot_individual
		if i2 < i1:
			print 'Second index must be greater than first in option --plot-individual.'
		plot_indiv(bounds, p[i1:i2], [guess, Delta_Ar], [guess_line_int[i1:i2], line_int[i1:i2]])
	
	# Plot the reddening profile on top of the stacked stellar probability densities
	if values.plotfn != None:
			sys.stderr.write('Plotting profile to %s ...\n' % values.plotfn)
	if (values.plotfn != None) or values.show:
		plot_profile(bounds, p, [guess, Delta_Ar, Delta_Ar_mean], values.plotfn, values.overplot)
	if values.show:
		plt.show()
	
	return 0


if __name__ == '__main__':
	main()

