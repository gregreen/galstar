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


from astroutils import equatorial2galactic
from galfast_utils import get_objects
import numpy as np
import pyfits
import argparse, sys
from scipy import interpolate
from os.path import abspath


def print_for_galstar(ra_dec, mags, errs, params):
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
	truth = ''
	count = 0
	for i,m in enumerate(mags):
		print_line = True
		line = '\n'
		truthline = ''
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
			for p in params[i]:
				truthline += '%f\t' % p
			truthline = truthline.rstrip('\t') + '\n'
		if print_line:
			outtxt += line
			truth += truthline
			count += 1
	outtxt += '\n'
	return count, outtxt, truth

def main():
	parser = argparse.ArgumentParser(prog='read_sdss',  description='Translate galfast output to ASCII input for galstar', add_help=True)
	parser.add_argument('inputs', nargs='+', type=str, help='FITS files')
	parser.add_argument('--filterr', type=float, default=None, help='Filter objects with errors greater than the specified amount')
	parser.add_argument('--maxmag', type=float, nargs='+', default=None, help='Maximum absolute magnitude to allow')
	parser.add_argument('--minmag', type=float, nargs='+', default=None, help='Minimum absolute magnitude to allow')
	parser.add_argument('--toscreen', action='store_true', help='Print results to screen, rather than outputting to ASCII files')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Set up the magnitude cutoffs
	mag_max = None
	if values.maxmag != None:
		if len(values.maxmag) == 1:
			mag_max = np.empty(5, dtype=float)
			mag_max.fill(values.maxmag)
		elif len(values.maxmag) == 5:
			mag_max = np.array(values.maxmag)
		else:
			print 'Invalid # of arguments for maxmag. Enter either a unform magnitude limit, or a separate limit for each band (5).'
			return 1
	mag_min = None
	if values.minmag != None:
		if len(values.minmag) == 1:
			mag_min = np.empty(5, dtype=float)
			mag_min.fill(values.minmag)
		elif len(values.maxmag) == 5:
			mag_min = np.array(values.minmag)
		else:
			print 'Invalid # of arguments for minmag. Enter either a unform magnitude limit, or a separate limit for each band (5).'
			return 1
	
	for fn in values.inputs:
		print 'Processing %s ...' % fn
		
		# Load data from FITS file
		ra_dec, mags, LSSTmags, absmags, errs, params = get_objects(fn)
		
		# Construct filters
		N = len(ra_dec)
		filterr = np.empty(N, dtype=bool)
		filterr.fill(True)
		filtmin = np.empty(N, dtype=bool)
		filtmin.fill(True)
		filtmax = np.empty(N, dtype=bool)
		filtmax.fill(True)
		
		# Stars with large errors
		if values.filterr != None:
			for i in range(N):
				filterr[i] = (errs[i,:].max() <= values.filterr)
		
		# Faint stars
		if values.minmag != None:
			for i in range(N):
				filtmin[i] = (mags[i] >= mag_min).all()
		
		# Bright stars
		if values.maxmag != None:
			for i in range(N):
				filtmax[i] = (mags[i] <= mag_max).all()
		
		print filtmax
		print filtmin
		print filterr
		
		# Combine and apply filters
		idx = np.logical_and(filterr, filtmin)
		idx = np.logical_and(idx, filtmax)
		print idx
		ra_dec = ra_dec[idx]
		mags = mags[idx]
		errs = errs[idx]
		params = params[idx]
		N_filt = len(ra_dec)
		
		# Produce input for galstar
		if N_filt != 0:
			count, output, truth = print_for_galstar(ra_dec, mags, errs, params)
			if values.toscreen:
				print output
			else:
				tmp = abspath(fn).split('.')
				output_fn = ''.join(tmp[:-1]) + '.in'
				f = open(output_fn, 'w')
				f.write(output)
				f.close()
				truth_fn = ''.join(tmp[:-1]) + '.truth'
				f = open(truth_fn, 'w')
				f.write(truth)
				f.close()
				print 'Written to %s' % output_fn
			print '-> %d objects (of %d total).' % (count, N)
		else:
			print 'No stars match specified criteria.'
	
	print 'Done.'
	
	return 0

if __name__ == '__main__':
	main()

