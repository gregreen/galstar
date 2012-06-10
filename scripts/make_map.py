#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       make_map.py
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

import numpy as np
import healpy as hp

import pyfits
import gzip

import sys, argparse
from os.path import abspath


def load_reddening(fname):
	'''
	Load the pixels contained in the file(s) given by <fname>,
	returning the reddening profile, along with the healpix
	pixel number and chi^2/d.o.f. of the fit for each line of
	sight.
	
	Input:
		fname		(string or list of strings)
	
	Output:
		mu_anchors	(list of flat np.float64 arrays)
		Ar_anchors	(list of flat np.float64 arrays)
		pix_index	(list of integers)
		chi2dof		(listo f floats)
	'''
	
	if type(fname) is str:
		fname = [fname]
	
	mu_anchors_list = []
	Ar_anchors_list = []
	pix_index_list = []
	chi2dof_list = []
	
	# Store (DM, Ar) fit for each healpix pixel
	for filename in fname:
		f = open(abspath(filename), 'rb')
		
		while True:
			try:
				pix_index = np.fromfile(f, dtype=np.uint64, count=1)[0]
				N_stars = np.fromfile(f, dtype=np.uint32, count=1)[0]
				measure = np.fromfile(f, dtype=np.float64, count=1)[0]
				success = np.fromfile(f, dtype=np.uint16, count=1)[0]
				N_regions = np.fromfile(f, dtype=np.uint16, count=1)[0]
				line_int = np.fromfile(f, dtype=np.float64, count=N_stars)
				mu_anchors = np.fromfile(f, dtype=np.float64, count=N_regions+1)
				Ar_anchors = np.fromfile(f, dtype=np.float64, count=N_regions+1)
				
				'''
				print 'pixel index: %d' % pix_index
				print '# of stars: %d' % N_stars
				print 'chisq: %.2f (%.2f / d.o.f.)' % (measure, measure / float(N_stars - int(N_regions) - 1))
				print 'success: %d' % success
				print '# of regions: %d' % N_regions
				print 'line integrals:'
				print line_int
				print 'mu anchors:'
				print mu_anchors
				print 'A_r anchors:'
				print Ar_anchors
				print ''
				'''
				
				mu_anchors_list.append(mu_anchors)
				Ar_anchors_list.append(Ar_anchors)
				pix_index_list.append(pix_index)
				chi2dof_list.append(measure / float(N_stars - int(N_regions) - 1))
				
			except:
				f.close()
				break
		
		f.close()
	
	return mu_anchors_list, Ar_anchors_list, np.array(pix_index_list, dtype=np.uint64), np.array(chi2dof_list, dtype=np.float64)


def eval_Ar(mu_anchors, Ar_anchors, pix_index, mu, nside=256):
	if type(mu_anchors) is not list:
		mu_anchors = [mu_anchors]
	if type(Ar_anchors) is not list:
		Ar_anchors = [Ar_anchors]
	if len(mu_anchors) != len(Ar_anchors):
		raise Exception('mu_anchors and Ar_anchors must contain same # of elements.')
	if len(mu_anchors) != len(pix_index):
		raise Exception('mu_anchors and pix_index must contain same # of elements.')
	
	# Create an empty map for each value of mu
	Ar_map = np.zeros((len(mu), 12*nside*nside), dtype=np.float64)
	
	# Sort the elements in mu
	if len(mu) == 0:
		mu = [mu]
	mu.sort()
	
	# Evaluate each pixel in the maps
	for i, (mu_arr, Ar_arr) in enumerate(zip(mu_anchors, Ar_anchors)):
		n = 0
		for j in xrange(1, len(mu_arr)):
			if mu_arr[j] >= mu[n]:
				slope = (mu_arr[j] - mu_arr[j-1]) / (Ar_arr[j] - Ar_arr[j-1])
				Ar_map[n, pix_index[i]] = Ar_arr[j-1] + slope * (mu[n] - mu_arr[j-1])
				n += 1
				if n >= len(mu):
					break
	
	return Ar_map


def main():
	parser = argparse.ArgumentParser(prog='make_map.py', description='Generate a map from the given reddening files generated by fit_pdfs.py.', add_help=True)
	parser.add_argument('input', type=str, nargs='+', help='Reddening files generated by fit_pdfs.py.')
	parser.add_argument('-o', '--output', type=str, help='Output filename (of type FITS) for reddening map.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	mu_anchors, Ar_anchors, pix_index, chi2dof = load_reddening(values.input)
	
	mu_eval = np.array([8., 12., 16.], dtype=np.float64)
	Ar_map = eval_Ar(mu_anchors, Ar_anchors, pix_index, mu_eval)
	
	#print Ar_map[0].shape
	
	hp.visufunc.mollview(map=None, nest=True)
	plt.show()
	
	#m = np.arange(hp.nside2npix(256))
	#hp.mollview(m, min=0, max=m.max(), title='Mollview RING', nest=False)
	
	return 0

if __name__ == '__main__':
	main()

