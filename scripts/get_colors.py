#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       untitled.py
#       
#       Copyright 2011 Greg <greg@greg-G53JW>
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


from pyfits import getdata
from os.path import abspath
import numpy as np


Aratio = np.array([1.8236, 1.4241, 1.0000, 0.7409, 0.5821])


# Return intrinsic (u-g, g-r, r-i, i-z, z-y) for the given stellar data d
def get_colors(d):
	m_ugrizy = d[9]		# Before errors are added in
	# Determine reddening
	Ar = d[7]
	Ax = np.empty(5, dtype=float)
	for i in range(5):
		Ax[i] = Ar * Aratio[i]
	# Determine distance modulus
	DM = d[3]
	# Get intrinsic ugrizy
	ugrizy = m_ugrizy[:5] - Ax
	ugrizy -= DM
	# Compute colors
	color = np.empty(4, dtype=float)
	for i in range(4):
		color[i] = ugrizy[i] - ugrizy[i+1]
	return color, red_ugrizy, ugrizy


# Return (DM, Ar, Mr, FeH) for the given stellar data d
def get_params(d):
	return d[3], d[7], d[4], d[6]


# Return the absolute magnitudes (u,g,r,i,z,y) for the entire catalog of stars
def get_abs_mags(fname):
	fn = abspath(fname)
	data = getdata(fn)
	N = len(data)
	abs_mags = np.empty((N,6), dtype=float)
	for i, star in enumerate(data):
		color, red_ugrizy, ugrizy = get_colors(d)
		abs_mags[i] = ugrizy
	return abs_mags


# Return the absolute magnitude of a star in each bandpass (ugriz), given its "true" LSST apparent magnitudes, and its parameters (DM, Ar, Mr, FeH)
def calc_abs_mag(LSST_mags, params):
	return LSST_mags - params[0] - params[1]*Aratio


def main():
	fn = raw_input('FITS filename: ')
	fn = abspath(fn)
	data = getdata(fn)
	
	quitnow = False
	while not quitnow:
		index = raw_input('Index: ')
		try:
			index = int(index)
			color, red_ugrizy, ugrizy = get_colors(data[index])
			DM, Ar, Mr, FeH = get_params(data[index])
			print '\tParams:', (DM, Ar, Mr, FeH)
			print '\tColors:', color
			print '\tReddened ugrizy:', red_ugrizy
		except:
			quitnow = True
	
	return 0

if __name__ == '__main__':
	main()

