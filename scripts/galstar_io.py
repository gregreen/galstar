#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       galstar_io.py
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

from os.path import abspath
import gzip

import numpy as np
import scipy.ndimage.filters as filters
import scipy.weave as weave


def load_stats(fname, selection=None):
	'''
	Load statistics on each star from galstar output.
	
	Input:
		fname - filename of galstar stats file
		selection - indices of stars to load. If None, all stars are loaded.
	
	Output:
		converged (numpy bool array) - whether or not each star converged
		ln_evidence (numpy float64 array) - ln(Z) for each star, up to a constant
		mean (numpy float64 array) - mean model parameters for each star
		cov (numpy float64 array) - model-parameter covariance matrix for each star
	'''
	
	f = open(abspath(fname), 'rb')
	
	# Read in header
	N_files = np.fromfile(f, dtype=np.uint32, count=1)[0]
	N_dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
	
	# Determine which stars to load
	indices = None
	if selection == None:
		indices = xrange(N_files)
	else:
		indices = selection
	N = len(indices)
	
	# Set up arrays to hold statistics
	converged = np.empty(N, dtype=np.bool)
	ln_evidence = np.empty(N, dtype=np.float64)
	mean = np.empty((N, N_dim), dtype=np.float64)
	cov = np.empty((N, N_dim*N_dim), dtype=np.float64)
	
	# Read in statistics one at a time
	block_size = 8 * (1 + 2 * N_dim * (N_dim + 1)) + 1
	offset = lambda index: 8 + block_size*index
	for i,k in enumerate(indices):
		# Seek to the position of the next selected star
		f.seek(offset(k), 0)
		
		# Read in this star
		converged[i] = np.fromfile(f, dtype=np.bool, count=1)[0]
		ln_evidence[i] = np.fromfile(f, dtype=np.bool, count=1)[0]
		mean[i] = np.fromfile(f, dtype=np.float64, count=N_dim)
		cov[i] = np.fromfile(f, dtype=np.float64, count=N_dim*N_dim)
		
		# Skip over raw data included about star
		tmp = np.fromfile(f, dtype=np.float64, count=N_dim*(N_dim+1))
		tmp = np.fromfile(f, dtype=np.uint64, count=1)
	
	f.close()
	
	# Reshape the covariance array
	cov.shape = (N, N_dim, N_dim)
	
	return converged, ln_evidence, mean, cov


def load_bins(fname, sparse=True, selection=None):
	'''
	Load binned probability density functions (pdfs) from a galstar bin output file (gzipped or uncompressed).
	
	Input:
		fname - filename of binned data
		sparse - True if pdfs are stored in sparse format (i.e. not as flat arrays)
		selection - indices of stars to load. If None, all stars are loaded.
	
	Output:
		bounds[4] = [x_min, x_max, y_min, y_max]
		bin_data (numpy float64 array) = p(n, x, y), where n is the index of the star, and x and y are the axes (DM and Ar, for example)
	'''
	
	if fname.endswith('.gz') or fname.endswith('.gzip'):
		if sparse:
			raise Exception('Cannot load sparsely stored files in gzip format.')
		return load_bins_gzip(fname, selection)
	else:
		if sparse:
			return load_bins_sparse(fname, selection)
		else:
			return load_bins_uncompressed(fname, selection)


def load_bins_uncompressed(fname, selection=None):
	'''
	Load binned probability density functions (pdfs) from an uncompressed galstar bin output file.
	
	Input:
		fname - filename of binned data
		selection - indices of stars to load. If None, all stars are loaded.
	
	Output:
		bounds[4] = [x_min, x_max, y_min, y_max]
		bin_data (numpy float64 array) = p(n, x, y), where n is the index of the star, and x and y are the axes (DM and Ar, for example)
	'''
	
	f = open(abspath(fname), 'rb')
	
	# Read in header
	N_files = np.fromfile(f, dtype=np.uint32, count=1)
	bin_width = np.fromfile(f, dtype=np.uint32, count=2)
	bin_min = np.fromfile(f, dtype=np.float64, count=2)
	bin_max = np.fromfile(f, dtype=np.float64, count=2)
	bin_dx = np.fromfile(f, dtype=np.float64, count=2)
	
	# Read in pdfs
	bin_data = None
	if selection == None:	# Read in all pdfs
		bin_data = np.fromfile(f, dtype=np.float64)
	else:					# Read in only selected pdfs
		N_files_sel = len(selection)
		N_pix = np.prod(bin_width)
		offset = lambda index: 60 + 8*N_pix*index
		bin_data = np.empty((N_files_sel, N_pix), dtype=np.float64)
		
		for i,k in enumerate(selection):
			if k >= N_files:
				raise Exception('selection contains at least one index (%d) greater than # of stars (%d) in bin file.' % (k, N_files))
			f.seek(offset(k), 0)
			bin_data[i] = np.fromfile(f, dtype=np.float64, count=N_pix)
	
	f.close()
	
	# Reshape bin data
	N_files_empirical = bin_data.size / bin_width[0] / bin_width[1]
	bin_data.shape = (N_files_empirical, bin_width[0], bin_width[1])
	
	# Create list containing bounds
	bounds = [bin_min[0], bin_max[0], bin_min[1], bin_max[1]]
	
	return bounds, bin_data


def load_bins_gzip(fname, selection=None):
	'''
	Load binned probability density functions (pdfs) from a gzipped galstar bin output file.
	
	Input:
		fname - filename of binned data
	
	Output:
		bounds[4] = [x_min, x_max, y_min, y_max]
		bin_data (numpy float64 array) = p(n, x, y), where n is the index of the star, and x and y are the axes (DM and Ar, for example)
	'''
	
	f_gzip = gzip.open(abspath(fname), 'rb')
	
	# Read in header
	f = f_gzip.read(60)
	N_files = np.fromstring(f[0:4], dtype=np.uint32, count=1)
	bin_width = np.fromstring(f[4:12], dtype=np.uint32, count=2)
	bin_min = np.fromstring(f[12:28], dtype=np.float64, count=2)
	bin_max = np.fromstring(f[28:44], dtype=np.float64, count=2)
	bin_dx = np.fromstring(f[44:60], dtype=np.float64, count=2)
	
	# Read in pdfs
	bin_data = None
	if selection == None:	# Read everything in at once
		f = f_gzip.read()
		f_gzip.close()
		bin_data = np.fromstring(f, dtype=np.float64)
	else:					# Read in only the selected stars
		N_files_sel = len(selection)
		N_pix = int(np.prod(bin_width))
		offset = lambda index: 60 + 8*N_pix*index
		bin_data = np.empty((N_files_sel, N_pix), dtype=np.float64)
		
		for i,k in enumerate(selection):
			if k >= N_files:
				raise Exception('selection contains indices greater than # of stars in bin file.')
			f_gzip.seek(offset(k), 0)
			f = f_gzip.read(8*N_pix)
			bin_data[i] = np.fromstring(f, dtype=np.float64)
		
		f_gzip.close()
	
	# Reshape bin data
	N_files_empirical = bin_data.size / bin_width[0] / bin_width[1]
	bin_data.shape = (N_files_empirical, bin_width[0], bin_width[1])
	
	# Create list containing bounds
	bounds = [bin_min[0], bin_max[0], bin_min[1], bin_max[1]]
	
	return bounds, bin_data


def load_bins_sparse(fname, selection=None):
	'''
	Load binned probability density functions (pdfs) from a sparse, uncompressed galstar bin output file.
	
	Input:
		fname - filename of binned data
		selection - indices of stars to load. If None, all stars are loaded.
	
	Output:
		bounds[4] = [x_min, x_max, y_min, y_max]
		bin_data (numpy float64 array) = p(n, x, y), where n is the index of the star, and x and y are the axes (DM and Ar, for example)
	'''
	
	# Read in header
	f = open(abspath(fname), 'rb')
	N_files = np.fromfile(f, dtype=np.uint32, count=1)[0]
	bin_width = np.fromfile(f, dtype=np.uint32, count=2)
	bin_min = np.fromfile(f, dtype=np.float64, count=2)
	bin_max = np.fromfile(f, dtype=np.float64, count=2)
	bin_dx = np.fromfile(f, dtype=np.float64, count=2)
	f.close()
	
	sel_sorted = None
	if selection == None:
		sel_sorted = np.arange(N_files, dtype=np.uint32)
	else:
		sel_sorted = np.sort(selection)
		if np.max(sel_sorted) > N_files:
			raise Exception('selection contains indices greater than # of stars in bin file.')
	
	# Create an empty array to populate
	N_files_sel = sel_sorted.size;
	bin_data = np.zeros((N_files_sel, bin_width[0], bin_width[1]), dtype=np.float64)
	
	# Load in all the stars
	code = """
		std::fstream infile(fname.c_str(), std::ios::binary | std::ios::in);
		
		unsigned int N_files;
		unsigned int width[2];
		double min[2];
		double max[2];
		double dx[2];
		infile.read(reinterpret_cast<char *>(&N_files), sizeof(unsigned int));
		infile.read(reinterpret_cast<char *>(&(width[0])), 2*sizeof(unsigned int));
		infile.read(reinterpret_cast<char *>(&(min[0])), 2*sizeof(double));
		infile.read(reinterpret_cast<char *>(&(max[0])), 2*sizeof(double));
		infile.read(reinterpret_cast<char *>(&(dx[0])), 2*sizeof(double));
		
		uint16_t i, j;
		double value;
		int m = 0;
		for(int n=0; n<(int)N_files; n++) {
			uint32_t N_nonzero;
			infile.read(reinterpret_cast<char*>(&N_nonzero), sizeof(uint32_t));
			//std::cout << "nonzero: " << N_nonzero << std::endl;
			
			if((int)sel_sorted(m) == n) {
				for(int k=0; k<(int)N_nonzero; k++) {
					infile.read(reinterpret_cast<char*>(&i), sizeof(i));
					infile.read(reinterpret_cast<char*>(&j), sizeof(j));
					infile.read(reinterpret_cast<char*>(&value), sizeof(value));
					
					if(((int)i < 0) || ((int)i > width[0]) || ((int)j < 0) || ((int)j > width[1])) {
						return_val = false;
						break;
					}
					bin_data((int)m, (int)i, (int)j) = value;
				}
				m++;
			} else {
				infile.seekg(12 * N_nonzero, std::ios::cur);
			}
			
			if(m >= N_files_sel) { break; }
		}
		
		infile.close();
		return_val = true;
	"""
	read_success = weave.inline(code, ['fname', 'bin_data', 'sel_sorted', 'N_files_sel'], headers=['<iostream>', '<fstream>', '<stdint.h>'], type_converters=weave.converters.blitz, compiler='gcc')
	
	if not read_success:
		raise Exception('Input file %s is corrupt.' % fname)
	
	# Create list containing bounds
	bounds = [bin_min[0], bin_max[0], bin_min[1], bin_max[1]]
	
	return bounds, bin_data


def smooth_bins(p, sigma):
	'''
	Smooth binned data with Gaussian kernel.
	
	Input:
		p(n, x, y), where n is the index of the star, and x and y are the axes (DM and Ar, for example)
		sigma = (sigma_x, sigma_y) - specifies the smoothing kernel to be applied
	
	Output:
		p_smooth(n, x, y) - p(n, x, y) smoothed with a gaussian kernel
	'''
	
	# Apply Gaussian smoothing to each image
	p_smooth = np.empty(p.shape, dtype=np.float64)
	filters.gaussian_filter(p, np.insert(sigma, 0, 0), output=p_smooth, mode='nearest')
	
	# Normalize each image to unit probability
	for i in xrange(p_smooth.shape[0]):
		p_smooth[i] /= np.sum(p_smooth[i])
	
	return p_smooth



def main():
	print 'galstar_io.py contains routines to load galstar output.'
	
	return 0

if __name__ == '__main__':
	main()

