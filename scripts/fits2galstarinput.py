#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       fits2galstarinput.py
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

from random import randint

import healpy as hp
import numpy as np
import pyfits
import tarfile


def lb2thetaphi(l, b):
	l_rad, b_rad = l * np.pi/180., b * np.pi/180.
	
	# Convert from Galactic to Cartesian coordinates
	sinb = np.sin(b_rad)
	x = np.cos(l_rad) * sinb
	y = np.sin(l_rad) * sinb
	z = np.cos(b_rad)
	
	# Convert from Cartesian to Spherical coordinates
	theta = np.arctan(np.divide(np.sqrt(x*x + y*y), z))
	phi = np.arctan2(y, x)
	
	return theta, phi


def main():
	parser = argparse.ArgumentParser(prog='fits2galstarinput.py', description='Generate galstar input files from LSD fits output.', add_help=True)
	parser.add_argument('FITS', type=str, help='FITS output from LSD.')
	parser.add_argument('tarout', type=str, help='Tarball output filename.')
	parser.add_argument('-pf', '--prefix', type=str, default='pix', help='Prefix for pixel names (default: pix).')
	parser.add_argument('-n', '--nside', type=int, default=128, help='healpix nside parameter (default: 128).')
	parser.add_argument('-r', '--ring', action='store_true', help='Use healpix ring ordering. If not specified, nested ordering is used.')
	parser.add_argument('-b', '--bounds', type=float, nargs=4, default=None, help='Restrict pixels to region enclosed by: l_min, l_max, b_min, b_max')
	parser.add_argument('-sp', '--split', type=int, default=1, help='Split into an arbitrary number of tarballs.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Load the stars from the FITS file
	d,h = pyfits.getdata(abspath(values.FITS), header=True)
	print 'Loaded %d stars.' % d.shape[0]
	
	# Convert (l, b) to spherical coordinates (physics convention)
	theta = np.pi/180. * (90. - d['b'])
	phi = np.pi/180. * d['l']
	
	# Convert spherical coordinates to healpix
	N_arr = hp.ang2pix(values.nside, theta, phi, nest=(not values.ring))
	
	# Get unique pixel numbers
	N_unique = np.unique(N_arr)
	print '%d unique healpix pixel(s) present.' % N_unique.size
	
	# Filter pixels by bounds
	if values.bounds != None:
		theta_0, phi_0 = hp.pix2ang(values.nside, N_unique, nest=(not values.ring))
		l_0 = 180./np.pi * phi_0
		b_0 = 90. - 180./np.pi * theta_0
		l_mask = np.logical_and((l_0 >= values.bounds[0]), (l_0 <= values.bounds[1]))
		b_mask = np.logical_and((b_0 >= values.bounds[2]), (b_0 <= values.bounds[3]))
		mask = np.logical_and(l_mask, b_mask)
		N_unique = N_unique[mask]
		print '%d unique healpix pixel(s) in bounds.' % N_unique.size
		if N_unique.size == 0:
			return 0
	
	# Open the tarball(s) which will gather all the output files
	tar = None
	if values.split < 1:
		print '--split must be positive.'
		return 1
	if values.split > 1:
		base = abspath(values.tarout).rstrip('.tar')
		tar = [tarfile.open('%s_%d.tar' % (base, i), 'w') for i in range(values.split)]
	else:
		tar = [tarfile.open(values.tarout, 'w')]
	
	# Generate .in file for each unique pixel number
	N_saved = 0
	N_stars_min = 1.e100
	N_stars_max = -1.
	for N in N_unique:
		# Get stars in this pixel
		mask = (N_arr == N)
		grizy = d['mean'][mask]
		err = d['err'][mask]
		outarr = np.hstack((grizy, err)).astype(np.float64)
		
		# Mask stars with nondetection or infinite variance in any bandpass
		mask_nan = np.isfinite(np.sum(err, axis=1))
		mask_nondetect = np.logical_not(np.sum((grizy == 0), axis=1).astype(np.bool))
		outarr = outarr[np.logical_and(mask_nan, mask_nondetect)]
		
		# Create binary .in file
		fname = abspath('%s_%d.in' % (values.prefix, N))
		f = open(fname, 'wb')
		
		# Write Header
		header_begin = np.array([np.mean(d['l'][mask]), np.mean(d['b'][mask])], dtype=np.float64)
		N_stars = np.array([outarr.shape[0]], np.uint32)
		f.write(header_begin.tostring())
		f.write(N_stars.tostring())
		
		# Write magnitudes and errors
		f.write(outarr.tostring())
		f.close()
		
		# Add the .in file to one of the tarballs
		dir_tmp, fname_short = os.path.split(fname)
		tar[randint(0,values.split-1)].add(fname, arcname=fname_short)
		os.remove(fname)
		
		# Record number of stars saved to pixel
		N_saved += outarr.shape[0]
		if outarr.shape[0] < N_stars_min:
			N_stars_min = outarr.shape[0]
		if outarr.shape[0] > N_stars_max:
			N_stars_max = outarr.shape[0]
	
	for t in tar:
		t.close()
	
	print 'Saved %d stars to %d galstar input file(s) (min: %d, max: %d, mean: %.1f).' % (N_saved, N_unique.size, N_stars_min, N_stars_max, float(N_saved)/float(N_unique.size))
	
	return 0

if __name__ == '__main__':
	main()

