#!/usr/bin/env python2.7
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
from matplotlib.ticker import MultipleLocator, MaxNLocator, NullFormatter, AutoMinorLocator, AutoLocator
from struct import unpack
from os.path import abspath
from operator import itemgetter
from math import sqrt

from galstar_io import *

DM, Ar, Mr, FeH = range(4)

'''
# This code is made obsolete by galstar_io.py
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
'''

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

def get_objects_ascii(txt_fname):
	f = open(txt_fname, 'r')
	params_list = []
	for line in f:
		tmp = line.lstrip().rstrip().split()
		if len(tmp) != 0:
			params_list.append(tmp)
	params = np.array(params_list, dtype=float)
	return params


def main():
	parser = argparse.ArgumentParser(prog='plot_stats.py', description='Plot overview of galstar output', add_help=True)
	parser.add_argument('files', type=str, nargs='+', help='Galstar statistics datafiles to open')
	parser.add_argument('-cnv', '--converged', action='store_true', help='Filter out nonconverged stars')
	parser.add_argument('-err', '--filterr', type=float, default=None, help='Filter out stars with errors greater than specified amount')
	parser.add_argument('-mag', '--filtmag', type=float, default=None, help='Filter out stars with magnitude greater than specified amount')
	#parser.add_argument('--galfast_comp', type=str, help='Galfast fits file')
	#parser.add_argument('--galfast_only', action='store_true', help='Plot only galfast catalog')
	#parser.add_argument('--txt_comp', type=str, help='Text file containing true stellar parameter values')
	parser.add_argument('-o', '--output', type=str, help='Output plot filename')
	parser.add_argument('-fig', '--figsize', type=float, nargs=2, default=(5,5), help='Figure size in inches (width, height)')
	parser.add_argument('-norm', '--normalize', nargs='+', type=str, default=None, help='Normalize DM or Ar to standard deviation')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=False)
	
	# Set up figure
	fig = plt.figure(figsize=values.figsize, dpi=300)
	fig.suptitle(r'$\mathrm{Scatter\ Plot\ of\ } ( \mu , A_r )$', fontsize=22)
	
	# Determine geometry of scatter plot and histograms
	scatter_left, scatter_bottom = 0.16, 0.1
	scatter_width, scatter_height = 0.62, 0.62
	buffer_x, buffer_y = 0., 0. #0.08, 0.05
	histx_height, histy_height = 0.17, 0.17
	rect_scatter = [scatter_left, scatter_bottom, scatter_width, scatter_height]
	rect_histx = [scatter_left, scatter_bottom+scatter_height+buffer_y, scatter_width, histx_height]
	rect_histy = [scatter_left+scatter_width+buffer_x, scatter_bottom, histy_height, scatter_height]
	
	# Set up the figure with a scatter plot and two histograms
	ax_scatter = fig.add_axes(rect_scatter)
	ax_histx = fig.add_axes(rect_histx)
	ax_histy = fig.add_axes(rect_histy)
	
	params = []
	if values.normalize != None:
		params = values.normalize
		for i,p in enumerate(params):
			params[i] = p.lower()
	if 'dm' in params:
		ax_scatter.set_xlabel(r'$\mu / \sigma_{\mu}$', fontsize=18)
	else:
		ax_scatter.set_xlabel(r'$\mu$', fontsize=18)
	if 'ar' in params:
		ax_scatter.set_ylabel(r'$A_r / \sigma_{A_r}$', fontsize=18)
	else:
		ax_scatter.set_ylabel(r'$A_r$', fontsize=18)
	
	# Info for histograms
	DM = []
	Ar = []
	
	# Handle each stats file separately
	for f in values.files:
		converged, ln_evidence, mean, cov = load_stats(f)
		
		# Apply filters
		idx = np.empty(converged.size, dtype=bool)
		idx.fill(True)
		if values.converged:
			idx = idx & (converged == True)
		if values.filtmag != None:
			idx = idx & np.all((mean < values.filtmag), axis=1)
		mean = mean[idx]
		
		# Normalize to errors
		if values.normalize != None:
			err = np.sqrt(np.diagonal(cov, axis1=1, axis2=2)[idx])
			if 'dm' not in params:
				err[:,0] = 1.
			if 'ar' not in params:
				err[:,1] = 1.
			mean = mean / err
		
		# Make scatterplot
		ax_scatter.plot(mean[:,0], mean[:,1], 'b.', linestyle='None', markersize=2)
		
		# Append information for histograms
		DM.append(mean[:,0])
		Ar.append(mean[:,1])
	
	# Plot histograms
	DM = np.hstack(DM)
	Ar = np.hstack(Ar)
	DM_sorted = np.sort(DM)
	Ar_sorted = np.sort(Ar)
	xmin, xmax = DM_sorted[int(0.02*(DM.size-1))], DM_sorted[int(0.98*(DM.size-1))]
	ymin, ymax = Ar_sorted[int(0.02*(Ar.size-1))], Ar_sorted[int(0.98*(Ar.size-1))]
	ymin = 0.
	#if ymin < 0.:
	#	ymin = 0.
	ax_scatter.set_xlim(xmin, xmax)
	ax_scatter.set_ylim(ymin, ymax)
	#xmin, xmax = ax_scatter.get_xlim()
	#ymin, ymax = ax_scatter.get_ylim()
	ax_histx.hist(np.hstack(DM), range=(xmin, xmax), bins=20, alpha=0.5, orientation='vertical')
	ax_histy.hist(np.hstack(Ar), range=(ymin, ymax), bins=20, alpha=0.5, orientation='horizontal')
	ax_histx.set_xlim(xmin, xmax)
	ax_histy.set_ylim(ymin, ymax)
	ax_histx.set_yticklabels([])
	ax_histy.set_xticklabels([])
	ax_histx.set_xticklabels([])
	ax_histy.set_yticklabels([])
	
	# Save plot to file
	if values.output != None:
		fn = abspath(values.output)
		if '.' not in fn:
			fn += '.png'
		fig.savefig(fn, dpi=300)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

