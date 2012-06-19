#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       query_lsd.py
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

import os, sys, argparse
from os.path import abspath

import healpy as hp
import numpy as np
import pyfits

import lsd

import iterators

import matplotlib.pyplot as plt


def mapper(qresult, nside, bounds):
	obj = lsd.colgroup.fromiter(qresult, blocks=True)
	
	if (obj != None) and (len(obj) > 0):
		# Determine healpix index of each star
		theta = np.pi/180. * (90. - obj['b'])
		phi = np.pi/180. * obj['l']
		pix_indices = hp.ang2pix(nside, theta, phi, nest=False)
		
		# Group together stars having same index
		for pix_index, block_indices in iterators.index_by_key(pix_indices):
			# Filter out pixels by bounds
			if bounds != None:
				theta_0, phi_0 = hp.pix2ang(nside, pix_index, nest=False)
				l_0 = 180./np.pi * phi_0
				b_0 = 90. - 180./np.pi * theta_0
				if (l_0 < bounds[0]) or (l_0 > bounds[1]) or (b_0 < bounds[2]) or (b_0 > bounds[3]):
					continue
			
			yield (pix_index, obj[block_indices])


def reducer(keyvalue):
	pix_index, obj = keyvalue
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	
	# Find stars with bad detections
	mask_zero_mag = (obj['mean'] == 0.)
	mask_zero_err = (obj['err'] == 0.)
	mask_nan_mag = np.isnan(obj['mean'])
	mask_nan_err = np.isnan(obj['err'])
	
	# Set errors for nondetections to some large number
	obj['mean'][mask_nan_mag] = 0.
	obj['err'][mask_zero_err] = 1.e10
	obj['err'][mask_nan_err] = 1.e10
	obj['err'][mask_zero_mag] = 1.e10
	
	# Combine and apply the masks
	mask_detect = np.sum(obj['mean'], axis=1).astype(np.bool)
	mask_informative = (np.sum(obj['err'] > 1.e10, axis=1) < 3).astype(np.bool)
	mask_keep = np.logical_and(mask_detect, mask_informative)
	
	yield (pix_index, obj[mask_keep])


def start_file(base_fname, index):
	fout = open('%s_%d.in' % (base_fname, index), 'wb')
	f.write(np.array([0], dtype=np.uint32).tostring())
	return f


def main():
	parser = argparse.ArgumentParser(prog='query_lsd.py', description='Generate galstar input files from PanSTARRS data.', add_help=True)
	parser.add_argument('out', type=str, help='Base filename for output.')
	parser.add_argument('-n', '--nside', type=int, default=512, help='healpix nside parameter (default: 512).')
	parser.add_argument('-b', '--bounds', type=float, nargs=4, default=None, help='Restrict pixels to region enclosed by: l_min, l_max, b_min, b_max.')
	parser.add_argument('-fs', '--filesize', type=int, default=85000, help='Number of stars per input file (default: 85000).')
	parser.add_argument('-min', '--min_stars', type=int, default=15, help='Minimum # of stars in pixel.')
	parser.add_argument('-vis', '--visualize', action='store_true', help='Show plot of footprint.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Determine the query bounds
	query_bounds = None
	if values.bounds != None:
		query_bounds = []
		query_bounds.append(0.)
		query_bounds.append(360.)
		pix_height = 90. / 2**np.sqrt(values.nside / 12)
		query_bounds.append(max(-90., values.bounds[2] - 5.*pix_height))
		query_bounds.append(min(90., values.bounds[3] + 5.*pix_height))
	else:
		query_bounds = [0., 360., -90., 90.]
	query_bounds = lsd.bounds.rectangle(query_bounds[0], query_bounds[2], query_bounds[1], query_bounds[3], coordsys='gal')
	query_bounds = lsd.bounds.make_canonical(query_bounds)
	
	# Set up the query
	db = lsd.DB(os.environ['LSD_DB'])
	query = "select obj_id, equgal(ra, dec) as (l, b), mean, err, mean_ap, nmag_ok from ucal_magsqv where (numpy.sum(nmag_ok > 0, axis=1) >= 4) & (nmag_ok[:,0] > 0) & (numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2)"
	query = db.query(query)
	
	# Determine the base output filename (without the '.in' extension)
	base_fname = abspath(values.out)
	if base_fname.endswith('.in'):
		base_fname = base_fname[:-3]
	
	# Keep track of number of stars saved to each file
	N_pix_used = []
	N_saved = []
	N_stars_min = 1.e100
	N_stars_max = -1.
	
	# Initialize map to store number of stars in each pixel
	pix_map = None
	if values.visualize:
		pix_map = np.zeros(12 * values.nside**2, dtype=np.uint64)
	
	f = None
	findex = 0
	N_stars_in_file = None
	N_pix_in_file = None
	
	N_pix_total = 0
	N_stars_total = 0
	l_min = 1.e100
	l_max = -1.e100
	b_min = 1.e100
	b_max = -1.e100
	
	# Save each pixel to the file with the least number of stars
	for (pix_index, obj) in query.execute([(mapper, values.nside, values.bounds), reducer], group_by_static_cell=True, bounds=query_bounds):
		N_pix_total += 1
		N_stars_total += len(obj)
		
		if len(obj) < values.min_stars:
			continue
		
		# Open new file, if necessary
		if f is None:
			f = open('%s_%d.in' % (base_fname, findex), 'wb')
			f.write(np.array([0], dtype=np.uint32).tostring())
			findex += 1
			N_stars_in_file = 0
			N_pix_in_file = 0
		
		# Create the output matrix
		outarr = np.hstack((obj['mean'], obj['err'])).astype(np.float64)
		
		# Write pixel header
		N_stars = np.array([outarr.shape[0]], dtype=np.uint32)
		gal_lb = np.array([np.mean(obj['l']), np.mean(obj['b'])], dtype=np.float64)
		if gal_lb[0] < l_min:
			l_min = gal_lb[0]
		if gal_lb[0] > l_max:
			l_max = gal_lb[0]
		if gal_lb[1] < b_min:
			b_min = gal_lb[1]
		if gal_lb[1] > b_max:
			b_max = gal_lb[1]
		f.write(np.array([pix_index], dtype=np.uint32).tostring())	# Pixel index	(uint32)
		f.write(gal_lb.tostring())									# (l, b)		(2 x float64)
		f.write(N_stars.tostring())									# N_stars		(uint32)
		
		# Write magnitudes and errors
		f.write(outarr.tostring())									# 5xmag, 5xerr	(10 x float64)
		
		# Record number of stars saved to pixel
		N_pix_in_file += 1
		N_stars_in_file += outarr.shape[0]
		if outarr.shape[0] < N_stars_min:
			N_stars_min = outarr.shape[0]
		if outarr.shape[0] > N_stars_max:
			N_stars_max = outarr.shape[0]
		if values.visualize:
			pix_map[N] = outarr.shape[0]
		
		# Close file, if number of stars saved to file exceeds specified number
		if N_stars_in_file >= values.filesize:
			f.seek(0)
			f.write(np.array([N_pix_in_file], dtype=np.uint32).tostring())
			f.close()
			f = None
			N_pix_used.append(N_pix_in_file)
			N_saved.append(N_stars_in_file)
	
	# Close file, if necessary
	if f != None:
		f.seek(0)
		f.write(np.array([N_pix_in_file], dtype=np.uint32).tostring())
		f.close()
		N_pix_used.append(N_pix_in_file)
		N_saved.append(N_stars_in_file)
	
	if sum(N_pix_used) != 0:
		print 'Saved %d stars from %d healpix pixels to %d galstar input file(s) (per pixel min: %d, max: %d, mean: %.1f).' % (sum(N_saved), sum(N_pix_used), findex, N_stars_min, N_stars_max, float(sum(N_saved))/float(sum(N_pix_used)))
		print '# of stars in footprint: %d.' % N_stars_total
		print '# of pixels in footprint: %d.' % N_pix_total
	else:
		print 'No pixels in specified bounds.'
	
	if (values.bounds != None) and (np.sum(N_pix_used) != 0):
		print ''
		print 'Bounds of included pixel centers:'
		print '\t(l_min, l_max) = (%.3f, %.3f)' % (l_min, l_max)
		print '\t(b_min, b_max) = (%.3f, %.3f)' % (b_min, b_max)
	
	# Show footprint of stored pixels on sky
	if values.visualize:
		hp.visufunc.mollview(map=np.log(pix_map), nest=True, title='Footprint', coord='G', xsize=5000)
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

