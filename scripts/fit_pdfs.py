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

# TODO:
#     1. Add command-line arguments
#     2. Read pdfs from .npz file (requires galstar to write .npz files)


import numpy as np
import scipy.ndimage.filters as filters
from scipy import weave
from scipy.optimize import leastsq
import gzip

import matplotlib as mplib
import matplotlib.pyplot as plt



# Load stats from file
def read_stats(fname_list):
	N_files = len(fname_list)
	
	# Set up data structures to house statistics
	mean = np.empty((N_files, 4), dtype=np.float64)
	cov = np.empty((N_files, 4, 4), dtype=np.float64)
	converged = np.empty(N_files, dtype=np.bool)
	
	# Extract stats from each file
	for i,fname in enumerate(fname_list):
		f = open(fname, 'rb')
		
		converged[i] = unpack('?', f.read(1))[0]			# Whether fit converged
		
		# Skip over Max. Likelihood information
		N_MLs = unpack('I', f.read(4))[0]					# Number of max. likelihoods to read
		for i in range(2*N_MLs):
			tmp = unpack('I', f.read(4))					# Dimension of this ML
			tmp = unpack('d', f.read(8))					# Pos. of max. likelihood along this axis
		
		# Skip # of dimensions in fit (which we know to be 4)
		tmp = f.read(4)
		
		# Read in mean
		mean[i] = np.fromfile(f, dtype=np.float64, count=4)
		
		# Read in covariance matrix (complicated because only the upper triangle is stored)
		tmp = np.fromfile(f, dtype=np.float64, count=10)
		cov[i, 0, :4] = tmp[:4]
		cov[i, 1, 0] = tmp[1]
		cov[i, 1, 1:4] = tmp[4:7]
		cov[i, 2, 0] = tmp[2]
		cov[i, 2, 1] = tmp[5]
		cov[i, 2, 2:4] = tmp[7:9]
		cov[i, 3, 0] = tmp[3]
		cov[i, 3, 1] = tmp[6]
		cov[i, 3, 2] = tmp[8]
		cov[i, 3, 3] = tmp[9]
		
		f.close()
		for i in range(4):
			for j in range(i):
				cov[i][j] = cov[j][i]
	
	return converged, mean, cov, ML_dim, ML


# Load bins from file
def load_bins(fname, is_log=True, return_log=True, min_p=None):
	x, y, p = np.loadtxt(fname, usecols=(0, 1, 2), unpack=True)
	
	# Filter out probabilities below cut
	if min_p != None:
		p[np.isnan(p)] = min_p
		p[(p < min_p)] = min_p
	
	# Transform scale (log or linear) of probabilities appropriately
	if is_log and not return_log:
		p = np.exp(p)
	elif not is_log and return_log:
		p = np.log(p)
	
	# Determine bounds along x- and y-axes
	bounds = [np.min(x), np.max(x), np.min(y), np.max(y)]
	# Reshape array
	dx = np.max(x[1:] - x[:-1])
	dy = np.max(y[1:] - y[:-1])
	Nx = round((bounds[1] - bounds[0]) / dx + 1.)
	Ny = round((bounds[3] - bounds[2]) / dy + 1.)
	p.shape = (Nx, Ny)
	
	# Expand bounds to cover edges of bins
	bounds[0] -= dx/2.
	bounds[1] += dx/2.
	bounds[2] -= dy/2.
	bounds[3] += dy/2.
	
	return bounds, p

def load_bins_stacked(fname_list, is_log=False, return_log=False, min_p=None):
	# Get information about coordinates, as well as number of pixels, from first file
	x, y, p_tmp = np.loadtxt(fname_list[0], usecols=(0, 1, 2), unpack=True)
	
	# Determine number of pixels
	N_pixels = p_tmp.size
	
	# Load all the images into one array
	N_files = len(fname_list)
	p = np.empty([N_files, N_pixels], dtype=p_tmp.dtype)
	for i in xrange(N_files):
		p[i] = np.loadtxt(fname_list[i], usecols=[2])
	
	# Filter out probabilities below cut
	if min_p != None:
		p[np.isnan(p)] = min_p
		p[(p < min_p)] = min_p
	
	# Transform scale (log or linear) of probabilities appropriately
	if is_log and not return_log:
		p = np.exp(p)
	elif not is_log and return_log:
		p = np.log(p)
	
	# Determine bounds along x- and y-axes
	bounds = [np.min(x), np.max(x), np.min(y), np.max(y)]
	# Reshape array
	dx = np.max(x[1:] - x[:-1])
	dy = np.max(y[1:] - y[:-1])
	Nx = round((bounds[1] - bounds[0]) / dx + 1.)
	Ny = round((bounds[3] - bounds[2]) / dy + 1.)
	p.shape = (N_files, Nx, Ny)
	
	# Expand bounds to cover edges of bins
	bounds[0] -= dx/2.
	bounds[1] += dx/2.
	bounds[2] -= dy/2.
	bounds[3] += dy/2.
	
	return bounds, p


def load_bins_binary(fname_list, is_log=False, return_log=False, min_p=None):
	bin_width = np.empty(2, dtype=np.uint32)
	bin_min = np.empty(2, dtype=np.float64)
	bin_max = np.empty(2, dtype=np.float64)
	bin_dx = np.empty(2, dtype=np.float64)
	bin_data = None
	
	for i,fname in enumerate(fname_list):
		f = open(fname, 'rb')
		
		# Read in header
		for j in xrange(2):
			bin_width[j] = np.fromfile(f, dtype=np.uint32, count=1)
			bin_min[j] = np.fromfile(f, dtype=np.float64, count=1)
			bin_max[j] = np.fromfile(f, dtype=np.float64, count=1)
			bin_dx[j] = np.fromfile(f, dtype=np.float64, count=1)
		
		# Read in binned pdf
		if i == 0:
			bin_data = np.empty([len(fname_list), bin_width[0]*bin_width[1]], dtype=np.float64)
		bin_data[i] = np.fromfile(f, dtype=np.float64, count=-1)
		
		f.close()
	
	bin_data.shape = (len(fname_list), bin_width[0], bin_width[1])
	bounds = [bin_min[0], bin_max[0], bin_min[1], bin_max[1]]
	
	return bounds, bin_data


def load_bins_binary_unified(fname):
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


def load_bins_binary_unified_gzip(fname):
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
def smooth_bins_stacked(p, sigma):
	# Apply Gaussian smoothing to each image
	p_smooth = np.empty(p.shape, dtype=np.float64)
	filters.gaussian_filter(p, np.insert(sigma, 0, 0), output=p_smooth, mode='nearest')
	
	# Normalize each image to unit probability
	for i in xrange(p_smooth.shape[0]):
		p_smooth[i] /= np.sum(p_smooth[i])
	
	return p_smooth


# Smooth binned data with Gaussian kernel
def smooth_bins(p, sigma):
	# Apply Gaussian smoothing
	p_smooth = np.empty(p.shape, dtype=np.float64)
	filters.gaussian_filter(p, sigma, output=p_smooth, mode='nearest')
	
	# Normalize to unit probability
	p_smooth /= np.sum(p_smooth)
	
	return p_smooth


def line_integral(y_anchors, img):
	# Determine the number of bins per piecewise linear region
	if img.shape[0] % (y_anchors.shape[0] - 1) != 0:
		raise Exception('Number of samples in mu not integer multiple of number of piecewise linear regions.')
	N_samples = img.shape[0] / (y_anchors.shape[0] - 1)
	
	tmp = 0.
	
	# Determine evaluation points in each region
	for i in xrange(len(y_anchors) - 1):
		# Determine the coordinates along the line to be evaluated
		x_eval = np.array(xrange(i*N_samples, (i+1)*N_samples), dtype=int)
		y_eval = np.linspace(y_anchors[i], y_anchors[i+1], N_samples+1)[1:]
		inbounds = (y_eval < img.shape[1]-1)
		x_eval = x_eval[inbounds]
		y_eval = y_eval[inbounds]
		
		# Determine the floor and ceiling indices in the y-direction
		y_floor = np.floor(y_eval)
		y_ceil = np.ceil(y_eval)
		ceil_diff = y_ceil - y_eval
		floor_diff = y_eval - y_floor
		ceil_indices = [x_eval.astype(int), y_ceil.astype(int)]
		floor_indices = [x_eval.astype(int), y_floor.astype(int)]
		
		# Evaluate the image along both indices
		ceil_line = img[ceil_indices]
		floor_line = img[floor_indices]
		
		# Weight each line according to its distance from the true coordinates (linear interpolation), and sum lines
		ceil_line *= floor_diff
		floor_line *= ceil_diff
		
		tmp += np.sum(ceil_line) + np.sum(floor_line)
	
	return tmp


def line_integral_weave(Delta_y, img):
	# Determine the number of bins per piecewise linear region
	if img.shape[0] % Delta_y.shape[0] != 0:
		raise Exception('Number of samples in mu not integer multiple of number of piecewise linear regions.')
	N_samples = img.shape[0] / Delta_y.shape[0]
	
	N_regions = Delta_y.shape[0]
	y_max = img.shape[1]
	
	code = """
		double tmp = 0.;
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
				tmp += (y_ceil - y) * img(x, (int)y_floor) + (y - y_floor) * img(x, (int)y_ceil);
			}
			if(y > y_max - 1) { continue; }
		}
		return_val = tmp;
	"""
	tmp = weave.inline(code, ['img', 'Delta_y', 'N_regions', 'N_samples', 'y_max'], type_converters=weave.converters.blitz, compiler='gcc')
	
	return tmp

# Compute the line integral through multiple images, stacked in <img>
def line_integral_stacked(Delta_y, img):
	# Determine the number of bins per piecewise linear region
	if img.shape[1] % Delta_y.shape[0] != 0:
		raise Exception('Number of samples in mu not integer multiple of number of piecewise linear regions.')
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
		

def chistacked(log_Delta_y, pdfs=None, chimax=5., regulator=10000.):
	Delta_y = np.exp(log_Delta_y)
	
	chi_tmp = np.sqrt(-2. * np.log(line_integral_stacked(Delta_y, pdfs)))
	chiscaled = chimax * np.tanh(chi_tmp / chimax)
	
	chiscaled += np.sum(Delta_y/regulator)
	
	return chiscaled


def minimize(pdfs, N_regions=5, chimax=5.):
	guess = np.log(np.random.ranf(N_regions) * 2.*150./float(N_regions))
	print 'guess:', guess
	x, success = leastsq(chistacked, guess, args=(pdfs, chimax), ftol=1.e-6, maxfev=10000)
	
	return x, success




def test_1():
	bounds, p = load_bins('/home/greg/projects/galstar/output/90_10/DM_Ar_0.txt', is_log=False, return_log=False)
	p_smooth = smooth_bins(p, [0,2])
	
	Ar_anchors = np.array([0,10,50,105,110,180])
	Delta_Ar = Ar_anchors[1:] - Ar_anchors[:-1]
	mu_anchors = np.linspace(0,150,Ar_anchors.size)
	print 'starting...'
	for i in xrange(100000):
		x = line_integral_weave(Delta_Ar, p_smooth)
	print 'done.'
	
	print line_integral(Ar_anchors, p_smooth)
	print line_integral_weave(Delta_Ar, p_smooth)
	print line_integral_stacked(Delta_Ar, np.array([p_smooth]))
	
	#print p_smooth[int(round((11.7-5.)/15.*150.)), int(round(0.28/5.*150.))], p_smooth[int(round(0.28/5.*150.)), int(round((11.7-5.)/15.*150.))], np.max(p_smooth)
	
	# Plot both points
	fig = plt.figure()
	
	ax1 = fig.add_subplot(2,1,1)
	ax1.imshow(p.T, extent=bounds, origin='lower', aspect='auto', cmap='hot')
	
	ax2 = fig.add_subplot(2,1,2)
	ax2.imshow(p_smooth.T, extent=bounds, origin='lower', aspect='auto', cmap='hot')
	
	ax2.plot(bounds[0]+mu_anchors*((bounds[1]-bounds[0])/150.), bounds[2]+Ar_anchors*((bounds[3]-bounds[2])/150.))
	ax2.set_xlim(bounds[0], bounds[1])
	ax2.set_ylim(bounds[2], bounds[3])
	
	plt.show()


def test_stacked():
	# Load pdfs
	fname_list = ['/home/greg/projects/galstar/output/90_10/DM_Ar_%d.txt' % i for i in xrange(20)]
	bounds, p = load_bins_stacked(fname_list)
	p_smooth = smooth_bins_stacked(p, [2,2])
	
	# Set up reddening profile
	Ar_anchors = np.array([0,10,50,105,110,180])
	Delta_Ar = Ar_anchors[1:] - Ar_anchors[:-1]
	mu_anchors = np.linspace(0,150,Ar_anchors.size)
	
	# Determine line integral for each pdf
	np.seterr(divide='ignore')
	print chistacked(Delta_Ar, pdfs=p_smooth)
	
	print 'Testing speed...'
	for i in xrange(10000):
		chistacked(Delta_Ar, pdfs=p_smooth)
	print 'Done.'
	
	fig = plt.figure()
	for i in xrange(8):
		ax = fig.add_subplot(4,2,i+1)
		img = np.log(p_smooth[i].T)
		img[np.isneginf(img)] = np.min(img[np.isfinite(img)])
		print img[0,0]
		ax.imshow(img, extent=bounds, origin='lower', aspect='auto', cmap='hot')
		ax.plot(bounds[0]+mu_anchors*((bounds[1]-bounds[0])/150.), bounds[2]+Ar_anchors*((bounds[3]-bounds[2])/150.))
		ax.set_xlim(bounds[0], bounds[1])
		ax.set_ylim(bounds[2], bounds[3])
	plt.show()


def test_fit():
	np.seterr(divide='ignore')
	
	# Load pdfs
	print 'Loading pdfs...'
	#fname_list = ['/home/greg/projects/galstar/output/90_10/DM_Ar_%d.dat' % i for i in xrange(4050)]
	#bounds, p = load_bins_binary(fname_list)
	fname = '/home/greg/projects/galstar/output/90_10/DM_Ar.dat.gz'
	bounds, p = load_bins_binary_unified_gzip(fname)
	p_smooth = smooth_bins_stacked(p, [2,2])
	print 'Done.'
	
	N_regions = 15
	
	# Fit reddening profile
	print 'Fitting reddening profile...'
	x, success = minimize(p_smooth, N_regions=N_regions, chimax=5.)
	print 'ln(Delta_Ar):', np.exp(x)
	print 'success:', success
	chi = chistacked(x, p_smooth, chimax=5.)
	print 'chi^2:', np.sum(chi*chi)
	
	# Overplot reddening profile on pdfs
	mu_anchors = np.linspace(0, 150, N_regions+1)
	Ar_anchors = np.empty(N_regions+1, dtype=x.dtype)
	for i in xrange(N_regions+1):
		Ar_anchors[i] = np.sum(np.exp(x[:i]))
	
	# Individual pdfs
	fig = plt.figure()
	for i in xrange(8):
		ax = fig.add_subplot(4,2,i+1)
		#img = np.log(p_smooth[i].T)
		#img[np.isneginf(img)] = np.min(img[np.isfinite(img)])
		img = p_smooth[i].T
		ax.imshow(img, extent=bounds, origin='lower', aspect='auto', cmap='hot')
		ax.plot(bounds[0]+mu_anchors*((bounds[1]-bounds[0])/150.), bounds[2]+Ar_anchors*((bounds[3]-bounds[2])/150.))
		ax.set_xlim(bounds[0], bounds[1])
		ax.set_ylim(bounds[2], bounds[3])
	
	# Stacked pdfs
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	img = np.average(p_smooth, axis=0).T
	img /= np.sum(img, axis=0)
	ax.imshow(img, extent=bounds, origin='lower', aspect='auto', cmap='hot')
	ax.plot(bounds[0]+mu_anchors*((bounds[1]-bounds[0])/150.), bounds[2]+Ar_anchors*((bounds[3]-bounds[2])/150.))
	ax.set_xlim(bounds[0], bounds[1])
	ax.set_ylim(bounds[2], bounds[3])
	
	plt.show()


def main():
	#bounds, p = load_bins_binary_unified('/home/greg/projects/galstar/output/90_10/DM_Ar.dat.bak')
	test_fit()
	
	return 0


if __name__ == '__main__':
	main()
