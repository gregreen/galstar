#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       read_galfast.py
#       
#       Copyright 2011 Gregory Green
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

# TODO:
# 	Implement filter option
# 	Determine errors: spline from photoerr files
# 	Write routine to compare galstar output to real values

from astroutils import equatorial2galactic
import numpy as np
import pyfits
import argparse, sys
from scipy import interpolate
from os.path import abspath

err_spline = None

def init_errs(photoerr_dir='/home/greg/projects/galstar/data'):
	global err_spline
	err_spline = []
	for bandpass in ['u','g','r','i','z']:
		photoerr_fn = 'SDSSugriz.SDSS%s.photoerr.txt' % bandpass
		if photoerr_dir[-1] == '/':
			f = open(photoerr_dir + photoerr_fn)
		else:
			f = open(photoerr_dir + '/' + photoerr_fn)
		mags, errs = [], []
		for l in f:
			line = l.lstrip().rstrip()
			if len(line) != 0:
				if line[0] != '#':
					m_tmp, e_tmp = line.split()[0:2]
					mags.append(m_tmp)
					errs.append(e_tmp)
		f.close()
		err_spline.append(interpolate.splrep(mags, errs))

def get_objects(galfast_fits_fname, filt=0.5):
	data = pyfits.getdata(galfast_fits_fname, ext=1)
	N = len(data)
	ra_dec = np.empty((N,2), dtype=float)
	mags = np.empty((N,5), dtype=float)
	errs = np.empty((N,5), dtype=float)
	params = np.empty((N,4), dtype=float)	# (DM, Ar, Mr, FeH)
	for i,d in enumerate(data):
		ra_dec[i] = d[1]
		mags[i] = d[11][:-1]	# Observed ugriz
		for j,m in enumerate(mags[i]):
			errs[i,j] = interpolate.splev(m, err_spline[j])
		params[i,0], params[i,1], params[i,2], params[i,3] = d[3], d[7], d[9][2], d[6]
	# Return [(RA, DEC),...] , [(u,g,r,i,z),...], [(sig_u,sig_g,...),...], [(DM,Ar,Mr,FeH),...]
	return ra_dec, mags, errs, params

def print_for_galstar(ra_dec, mags, errs):
	N = len(ra_dec)
	# Determine mean l, b
	avg_l, avg_b = 0., 0.
	for i in range(N):
		l,b = equatorial2galactic(ra_dec[i][0], ra_dec[i][1])
		avg_l += l
		avg_b += b
	avg_l /= N
	avg_b /= N
	outtxt = '%.3f\t%.3f' % (avg_l, avg_b)
	count = 0
	for i,m in enumerate(mags):
		print_line = True
		line = '\n'
		for j,m_X in enumerate(m):
			if m_X == float('inf'):
				print_line = False
				break
			if j != 0:
				line += '\t'
			line += '%.3f' % m_X
		if print_line:
			for s in errs[i]:
				line += '\t%.3f' % s
		if print_line:
			outtxt += line
			count += 1
	outtxt += '\n'
	return count, outtxt

def main():
	parser = argparse.ArgumentParser(prog='read_sdss',  description='Translate galfast output to ASCII input for galstar', add_help=True)
	parser.add_argument('inputs', nargs='+', type=str, help='FITS files')
	parser.add_argument('--filter', dest='filt', type=float, default=0.5, help='Filter objects with errors greater than the specified amount')
	parser.add_argument('--toscreen', action='store_true', help='Print results to screen, rather than outputting to ASCII files')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	init_errs()
	
	for fn in values.inputs:
		print 'Processing %s ...' % fn
		ra_dec, mags, errs, params = get_objects(fn, filt=values.filt)
		count, output = print_for_galstar(ra_dec, mags, errs)
		print '-> %d objects (of %d total).' % (count, len(ra_dec))
		if values.toscreen:
			print output
		else:
			tmp = abspath(fn).split('.')
			output_fn = ''.join(tmp[:-1]) + '.in'
			f = open(output_fn, 'w')
			f.write(output)
			f.close()
	
	print 'Done.'
	
	return 0

if __name__ == '__main__':
	main()

