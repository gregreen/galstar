#!/usr/bin/env python
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


import sys, argparse
from os.path import abspath

import healpy as hp
import numpy as np
import pyfits


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
	parser.add_argument('FITS', type=str, help="FITS output from LSD.")
	parser.add_argument('output', type=str, help='Base output filename (without extension)')
	parser.add_argument('-n', '--nside', type=int, default=128, help='healpix nside parameter (default: 32).')
	parser.add_argument('-r', '--ring', action='store_true', help='Use healpix ring ordering. If not specified, nested ordering is used.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	
	# Load the stars from the FITS file
	d,h = pyfits.getdata(abspath(values.FITS), header=True)
	print 'Loaded %d stars.' % d.shape[0]
	
	# Convert (l, b) to spherical coordinates (physics convention)
	#theta, phi = lb2thetaphi(d['l'], d['b'])
	theta = np.pi/180. * (90. - d['b'])
	phi = np.pi/180. * d['l']
	
	# Convert spherical coordinates to healpix
	N_arr = hp.ang2pix(values.nside, theta, phi, nest=(not values.ring))
	print N_arr.shape
	
	# Generate .in file for each unique pixel number
	N_saved = 0
	N_unique = np.unique(N_arr)
	print '%d unique healpix pixel(s) present.' % N_unique.size
	for N in N_unique:
		# Get stars in this pixel
		mask = (N_arr == N)
		grizy = d['mean'][mask]
		err = d['err'][mask]
		outarr = np.hstack((grizy, err))
		
		# Mask stars with nondetection or infinite variance in any bandpass
		mask_nan = np.isfinite(np.sum(err, axis=1))
		mask_nondetect = np.logical_not(np.sum((grizy == 0), axis=1).astype(np.bool))
		outarr = outarr[np.logical_and(mask_nan, mask_nondetect)]
		
		# Output mags and errs to a .in file
		N_saved += outarr.shape[0]
		fname = abspath('%s_%d.in' % (values.output, N))
		np.savetxt(fname, outarr, fmt='%.5f')
		
		# Prepend the average l, b to the file
		l_avg = np.mean(d['l'][mask])
		b_avg = np.mean(d['b'][mask])
		f = open(fname, 'r')
		ftxt = f.read()
		f.close()
		f = open(fname, 'w')
		ftxt = '%.3f %.3f\n%s' % (l_avg, b_avg, ftxt)
		f.write(ftxt)
		f.close()
	
	print 'Saved %d stars to %d galstar input file(s).' % (N_saved, N_unique.size)
	
	return 0

if __name__ == '__main__':
	main()

