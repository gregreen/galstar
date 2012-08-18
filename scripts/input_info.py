#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       input_info.py
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
import sys, argparse
from os.path import abspath


def seek_to_pixel(f, index):
	'''
	Seek to the beginning of the data (i.e. past header) in pixel #index,
	returning the header for the given pixel.
	
	Inputs:
		f (open 'rb' file object)	open galstar input file
		index						index of pixel to seek to
	
	Outputs:
		pix_index (uint32)		healpix index of pixel
		gal_lb (2 x float64)	galactic (l, b)
		N_stars (uint32)		# of stars in pixel
	'''
	if index < 0:
		raise ValueError('Pixel index must be non-negative.')
	
	try:
		f.seek(0, 0)
	except:
		raise Exception('Cannot seek in file <f>. The file may not be open.')
	npix = np.fromfile(f, dtype=np.uint32, count=1)[0]
	if index >= npix:
		raise Exception('Not enough pixels in input file.')
	
	pix_index, gal_lb, N_stars = None, None, None
	for i in xrange(index+1):
		pix_index = np.fromfile(f, dtype=np.uint32, count=1)[0]
		gal_lb = np.fromfile(f, dtype=np.float64, count=2)
		N_stars = np.fromfile(f, dtype=np.uint32, count=1)[0]
		if i != index:
			f.seek(N_stars * 13 * 8, 1)
	
	return pix_index, gal_lb, N_stars

def main():
	parser = argparse.ArgumentParser(prog='input_info.py', description='Print information about given galstar input file.', add_help=True)
	parser.add_argument('infile', type=str, help='galstar input file.')
	parser.add_argument('--npix', action='store_true', help='Print # of pixels in the given input file.')
	parser.add_argument('--pix_index', type=int, help='Print healpix index of the specified pixel.')
	parser.add_argument('--nstars', type=int, help='Print # of stars in the specified pixel.') 
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	f = open(abspath(values.infile), 'rb')
	npix = np.fromfile(f, dtype=np.uint32, count=1)[0]
	
	if values.npix:
		print npix
	
	if values.pix_index != None:
		pix_index, gal_lb, N_stars = seek_to_pixel(f, values.pix_index)
		print pix_index
	
	if values.nstars != None:
		pix_index, gal_lb, N_stars = seek_to_pixel(f, values.nstars)
		print N_stars
	
	f.close()
	
	return 0

if __name__ == '__main__':
	main()

