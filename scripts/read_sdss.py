#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       read_sdss.py
#       
#       Copyright 2011 Gregory <greg@greg-G53JW>
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


import pyfits as pf
import numpy as np
from math import log10, acos, asin, cos, sin, pi, atan2, log, sqrt
import sys, argparse


def get_objects(sdss_fits_fname, filt=0.5):
	f = pf.open(sdss_fits_fname)
	scidata = f[1].data
	N_obj = len(scidata)
	ra_dec = []
	mags = []
	errs = []
	norm = 2.5/log(10.)
	ln10 = log(10.)
	for obj in scidata:
		# Get (RA, DEC)
		ra_dec_tmp = np.empty(2, dtype=float)
		ra_dec_tmp[0], ra_dec_tmp[1] = obj[17:19]
		ra_dec.append(ra_dec_tmp)
		# Get AB Magnitudes (u,g,r,i,z)
		maggies, maggies_ivar = obj[22:24]
		mags_tmp, errs_tmp = np.empty(5, dtype=float), np.empty(5, dtype=float)
		AB_offset = [-0.04, 0., 0., 0., 0.02]
		use_obj = True
		for i,m in enumerate(maggies):
			if m > 0:
				sigma_m = 1./sqrt(maggies_ivar[i])
				if sigma_m/m > filt:
					use_obj = False
					break
				bias = 5./ln10 * (0.25*(sigma_m/m)**2. + 0.375*(sigma_m/m)**4.)
				mags_tmp[i] = 22.5 - 2.5*log10(m) + AB_offset[i] + bias
				errs_tmp[i] = 2.5/ln10 * sigma_m/m
				#print m, sigma_m, mags_tmp[i], errs_tmp[i], bias
			else:
				mags_tmp[i] = float('inf')
				errs_tmp[i] = float('inf')
		if use_obj:
			mags.append(mags_tmp)
			errs.append(errs_tmp)
	# Return [(RA, DEC),...] , [(u,g,r,i,z),...]
	return ra_dec, mags, errs, len(scidata)

def print_objects(ra_dec, mags, use_lb=False, filter_inf=False, include_err=False, std_err=0.05):
	if use_lb:
		header = '# l  \tb  '
	else:
		header = '# RA \tDEC  '
	header += '\tu  \tg  \tr  \ti  \tz'
	if include_err:
		header += '\tu_err\tg_err\tr_err\ti_err\tz_err'
	print header
	for i in range(len(ra_dec)):
		print_line = True
		if use_lb:
			l,b = equatorial2galactic(ra_dec[i][0], ra_dec[i][1])
			outtxt = '%.3f\t%.3f ' % (l, b)
		else:
			outtxt = '%.3f\t%.3f ' % (ra_dec[i][0], ra_dec[i][1])
		for m in mags[i]:
			if filter_inf and (m == float('inf')):
				print_line = False
				break
			else:
				outtxt += '\t%.3f' % m
		if print_line:
			if include_err:
				for j in range(5):
					outtxt += '\t%.3f' % std_err
			print outtxt

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
	return count, outtxt

def deg2rad(x):
	return x*pi/180.

def rad2deg(x):
	return x*180./pi

def equatorial2galactic(alpha, delta):
	a = deg2rad(alpha)
	d = deg2rad(delta)
	a_NGP = deg2rad(192.8595)	# RA of NGP
	d_NGP = deg2rad(27.12825)	# DEC of NGP
	l_NCP = deg2rad(123.932)	# l of NCP
	
	sinb = sin(d_NGP)*sin(d) + cos(d_NGP)*cos(d)*cos(a-a_NGP)
	b = asin(sinb)
	if (b < -pi/2.) or (b > pi/2.):	# Ensure that b is in the correct quadrant
		b = pi - b
	cosb = cos(b)
	y = cos(d)*sin(a-a_NGP)/cosb
	x = (cos(d_NGP)*sin(d) - sin(d_NGP)*cos(d)*cos(a-a_NGP))/cosb
	l = l_NCP - atan2(y,x)
	# Test if everything has worked out
	if (abs(cosb*sin(l_NCP-l) - y*cosb) > 0.01) or (abs(cosb*cos(l_NCP-l) - x*cosb) > 0.01):
		print "ERROR!!!"
	return rad2deg(l), rad2deg(b)

def main():
	parser = argparse.ArgumentParser(prog='read_sdss',  description='Translate FITS files of SDSS objects to ASCII input for galstar', add_help=True)
	parser.add_argument('inputs', nargs='+', type=str, help='FITS files')
	parser.add_argument('--filter', dest='filt', type=float, default=0.5, help='Filter objects with errors greater than the specified amount')
	parser.add_argument('--toscreen', action='store_true', help='Print results to screen, rather than outputting to ASCII files')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	for fn in values.inputs:
		print 'Processing %s ...' % fn
		ra_dec, mags, errs, N_tot = get_objects(fn, filt=values.filt)
		count, output = print_for_galstar(ra_dec, mags, errs)
		print '-> %d objects (of %d total).' % (count, N_tot)
		if values.toscreen:
			print output
		else:
			tmp = fn.split('.')
			output_fn = ''.join(tmp[:-1]) + '.in'
			f = open(output_fn, 'w')
			f.write(output)
			f.close()
	
	print 'Done.'
	
	return 0

if __name__ == '__main__':
	main()

