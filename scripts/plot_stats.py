#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       plot_stats.py
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


import sys, argparse
import matplotlib as mplib
import matplotlib.pyplot as plt
import numpy as np
import pyfits
from scipy import interpolate
from matplotlib.ticker import MultipleLocator, MaxNLocator
from struct import unpack
from os.path import abspath
from operator import itemgetter
from math import sqrt


DM, Ar, Mr, FeH = range(4)


def read_stats(fname):
	f = open(fname, 'rb')
	mean = np.empty(4, dtype=float)
	cov = np.empty((4, 4), dtype=float)
	converged = unpack('?', f.read(1))[0]				# Whether fit converged
	N_MLs = unpack('I', f.read(4))[0]					# Number of max. likelihoods to read
	ML_dim = []
	ML = []
	for i in range(N_MLs):
		ML_dim.append(np.empty(2, dtype=int))
		ML.append(np.empty(2, dtype=float))
		for k in range(2):
			ML_dim[i][k] = unpack('I', f.read(4))[0]	# Dimension of this ML
			ML[i][k] = unpack('d', f.read(8))[0]		# Pos. of max. likelihood along this axis
	tmp = f.read(4)										# Skip # of dimensions in fit (which we know to be 4)
	for i in range(4):
		mean[i] = unpack('d', f.read(8))[0]				# Read in means
	for i in range(4):
		for j in range(i, 4):
			cov[i][j] = unpack('d', f.read(8))[0]		# Read in covariance
	f.close()
	for i in range(4):
		for j in range(i):
			cov[i][j] = cov[j][i]
	return converged, mean, cov, ML_dim, ML

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
	if err_spline == None:
		init_errs()
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
		params[i,DM], params[i,Ar], params[i,Mr], params[i,FeH] = d[3], d[7], d[9][2], d[6]
	# Return [(RA, DEC),...] , [(u,g,r,i,z),...], [(sig_u,sig_g,...),...], [(DM,Ar,Mr,FeH),...]
	return ra_dec, mags, errs, params


def main():
	parser = argparse.ArgumentParser(prog='plot_stats.py', description='Plot overview of galstar output', add_help=True)
	parser.add_argument('files', type=str, nargs='+', help='Galstar statistics datafiles to open')
	parser.add_argument('--converged', action='store_true', help='Filter out nonconverged stars')
	parser.add_argument('--filterr', type=float, help='Filter out stars with errors greater than specified amount')
	parser.add_argument('--filtmag', type=float, help='Filter out stars with magnitude greater than specified amount')
	parser.add_argument('--galfast_comp', type=str, help='Galfast fits file')
	parser.add_argument('--galfast_only', action='store_true', help='Plot only galfast catalog')
	parser.add_argument('--output', type=str, help='Output plot filename')
	parser.add_argument('--errorbars', action='store_true', help='Show error bars on plots')
	parser.add_argument('--useML', type=int, default=-1, help='Index of max. likelihood in stats file to use')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Sort filenames
	files = values.files
	z = [(ff, int((ff.split('_')[-1]).split('.')[0])) for ff in files]
	z.sort(key=itemgetter(1))
	for n in range(len(z)):
		files[n] = z[n][0]
	
	# Read in means, covariances
	N = len(files)
	means = np.empty((N, 4), dtype=float)
	cov = np.empty((N, 4, 4), dtype=float)
	converged = np.empty(N, dtype=bool)
	ML = np.empty((N, 2), dtype=float)
	for i,fn in enumerate(files):
		converged[i], means[i], cov[i], tmp1, tmp2 = read_stats(abspath(fn))
		if values.useML != -1:
			ML[i] = tmp2[values.useML]
	
	# Initialize filter
	idx = np.empty(N, dtype=bool)
	idx.fill(True)
	
	# Calculate filter for nonconverged stars
	conv = np.empty(N, dtype=bool)
	conv.fill(True)
	if values.converged:
		conv = (converged == True)
	
	# Calculate filters related to galfast output
	ra_dec, mags, errs, params = None, None, None, None
	filterr = np.empty(N, dtype=bool)
	filterr.fill(True)
	filtmag = np.empty(N, dtype=bool)
	filtmag.fill(True)
	if values.galfast_comp:
		ra_dec, mags, errs, params = get_objects(abspath(values.galfast_comp))
		# Stars with large errors
		if values.filterr != None:
			for i in range(N):
				filterr[i] = (errs[i,:].max() <= values.filterr)
		# Faint stars
		if values.filtmag != None:
			for i in range(N):
				filtmag[i] = (mags[i,:].max() <= values.filtmag)
	
	# Combine and apply filters
	idx = np.logical_and(conv, filterr, filtmag)
	means = means[idx]
	cov = cov[idx]
	ML = ML[idx]
	N = len(means)
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	# Scatter plot of (DM, Ar)
	if values.galfast_comp == None:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		xerr, yerr = np.empty(N, dtype=float), np.empty(N, dtype=float)
		for i in range(N):
			xerr[i] = sqrt(cov[i,DM,DM])
			yerr[i] = sqrt(cov[i,Ar,Ar])
		if values.useML != -1:
			x = ML[:,0]
			y = ML[:,1]
		else:
			x = means[:,DM]
			y = means[:,Ar]
		if values.errorbars:
			ax.errorbar(x, y, xerr, yerr, linestyle='None')
		else:
			ax.plot(x, y, '.', linestyle='None', markersize=1)
		ax.set_xlabel(r'$\mu$', fontsize=18)
		ax.set_ylabel(r'$A_r$', fontsize=18)
		ax.set_title(r'$\mathrm{Scatter\ Plot\ of\ } ( \mu , A_r )$', fontsize=22)
		ax.set_ylim(0., ax.get_ylim()[1])
	elif values.galfast_only:	# Only plot galfast input
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		xerr, yerr = np.empty(N, dtype=float), np.empty(N, dtype=float)
		for i in range(N):
			xerr[i] = sqrt(cov[i,DM,DM])
			yerr[i] = sqrt(cov[i,Ar,Ar])
		params = params[idx]				# Apply same filter to galfast catalog as to galstar output
		x, y = None, None
		x = params[:,DM]
		y = params[:,Ar]
		if values.errorbars:
			ax.errorbar(x, y, xerr, yerr, linestyle='None')
		else:
			ax.plot(x, y, '.', linestyle='None', markersize=1)
		ax.set_xlabel(r'$\mu$', fontsize=18)
		ax.set_ylabel(r'$A_r$', fontsize=18)
		ax.set_title(r'$\mathrm{Galfast\ Catalog}$', fontsize=22)
	else:	# Compare with galfast input
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		xerr, yerr = np.empty(N, dtype=float), np.empty(N, dtype=float)
		for i in range(N):
			xerr[i] = sqrt(cov[i,DM,DM])
			yerr[i] = sqrt(cov[i,Ar,Ar])
		params = params[idx]				# Apply same filter to galfast catalog as to galstar output
		x, y = None, None
		if values.useML != -1:
			x = ML[:,0]
			y = ML[:,1]
		else:
			x = means[:,DM]
			y = means[:,Ar]
		for i in range(N):
			print params[i,DM], params[i,Ar], x[i], y[i]
		x -= params[:,DM]
		y -= params[:,Ar]
		if values.errorbars:
			ax.errorbar(x, y, xerr, yerr, linestyle='None')
		else:
			ax.plot(x, y, '.', linestyle='None', markersize=1)
		ax.set_xlabel(r'$\Delta \mu$', fontsize=18)
		ax.set_ylabel(r'$\Delta A_r$', fontsize=18)
		if values.useML != -1:
			ax.set_title(r'$\mathrm{Comparison\ with\ Galfast\ Catalog\ (ML)}$', fontsize=22)
		else:
			ax.set_title(r'$\mathrm{Comparison\ with\ Galfast\ Catalog\ (Means)}$', fontsize=22)
	
	# Save plot to file
	if values.output != None:
		fn = abspath(values.output)
		if '.' not in fn:
			fn += '.png'
		fig.savefig(fn)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

