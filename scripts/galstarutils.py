#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       galstarutils.py
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


import numpy as np
import pyfits
import sys
from scipy import interpolate
from struct import unpack
from os.path import abspath
from operator import itemgetter
from math import sqrt, isnan

import matplotlib.pyplot as plt;
import matplotlib as mplib


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
	DM, Ar, Mr, FeH = range(4)
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

# Sort filenames named according # in scheme *_#.*
def sort_filenames(fn_list):
	if len(fn_list) == 1:
		return fn_list
	z = [(ff, int((ff.split('_')[-1]).split('.')[0])) for ff in fn_list]
	z.sort(key=itemgetter(1))
	sorted_fn_list = []
	for n in range(len(z)):
		sorted_fn_list.append(z[n][0])
	return sorted_fn_list

def load_stacked(fn_list, stack_linear=False, normalize=False):
	N = len(fn_list)
	
	# Load the first pdf
	img, x, y, p = loadimage(fn_list[0])
	x_len, y_len = len(x), len(y)
	if stack_linear:
		img = np.exp(img)
	
	# Stack the remaining pdfs onto this one
	maxticks = 50.
	progress = 1./float(N)
	ticks = 0
	sys.stdout.write('Stacking pdfs: '); sys.stdout.flush()
	for i in range(1,N):
		# Update the progress bar
		progress += 1./float(N)
		while ticks < progress*maxticks:
			sys.stdout.write('#'); sys.stdout.flush()
			ticks += 1
		# Load and stack the current pdf
		if (len(x) == x_len) and (len(y) == y_len):
			img_tmp, x, y, p_tmp = loadimage(fn_list[i])
			idx = np.isnan(img_tmp)
			img_tmp[idx] = 0.
			if stack_linear:
				img_tmp = np.exp(img_tmp)
			img += img_tmp
			p += p_tmp
		else:
			print 'Error: Image #%d of has different dimensions than first image' % i
			return False
	sys.stdout.write('\n'); sys.stdout.flush()
	# Normalize probability densities to peak and return
	if stack_linear and normalize:
		for i in range(len(img)):
			print np.max(img[i,:])
			img[i,:] /= np.max(img[i,:])
	elif (not stack_linear) and normalize:
		for i in range(len(img)):
			min_i = np.min(img[i,:])
			max_i = np.max(img[i,:])
			img[i,:] -= max_i
			if min_i != max_i:
				img[i,:] *= min_i / (min_i - max_i)
	elif not stack_linear:
		img -= np.max(img)
	else:
		img /= np.max(img)
	p -= np.max(p)
	return img, x, y, p

# Load true parameter values from ascii file
def load_true_values(fn):
	params = np.loadtxt(fn, usecols=(0,1,2,3))
	return params

# Load SM-formatted image and return an array
def loadimage(fn):
	x, y, p = np.loadtxt(fn, usecols=(0, 1, 2), unpack=True)
	# Determine minumum nonzero probability
	ln_minp = np.nanmin(p)
	#ln_minp = p[0]
	#for p_i in p:
	#	if (not isnan(p_i)) and (p_i < ln_minp):
	#		ln_minp = p_i
	# Replace points with zero probability with minimum nonzero probability
	idx = np.isnan(p)
	p[idx] = ln_minp
	#for i,p_i in enumerate(p):
	#	if isnan(p_i): p[i] = ln_minp
	# Sort x and y and get dx and dy
	xs = x.copy(); xs.sort(); dx = xs[1:xs.size] - xs[0:xs.size-1]; dx = dx.max();
	ys = y.copy(); ys.sort(); dy = ys[1:ys.size] - ys[0:ys.size-1]; dy = dy.max();
	# Nearest-neighbor interpolation
	i = ((x - xs[0]) / dx).round().astype(int)
	j = ((y - ys[0]) / dy).round().astype(int)
	# Determine width and height of image
	nx = i.max() + 1
	ny = j.max() + 1
	# Fill in the image
	img = np.zeros([nx, ny]) #; img[:,:] = p.min();
	img[i,j] = p
	return img, x, y, p

def plotimg(img, x, y, ax, axis_labels=None, xlim=(None,None), ylim=(None,None), params=None, vmin=None):
	bounds = [x.min(), x.max(), y.min(), y.max()]
	for i in range(2):
		if xlim[i] != None: bounds[i] = xlim[i]
		if ylim[i] != None: bounds[i+2]= ylim[i]
	ax.imshow(img.transpose(), origin='lower', aspect='auto', interpolation='nearest', cmap='hot', extent=(x.min(),x.max(),y.min(),y.max()), vmin=vmin)
	# Set canvas size
	ax.set_xlim(bounds[0:2])
	ax.set_ylim(bounds[2:])
	# Set axis labels
	if axis_labels != None:
		ax.set_xlabel(r'$\mathrm{%s}$'%axis_labels[0], fontsize=20)
		ax.set_ylabel(r'$\mathrm{%s}$'%axis_labels[1], fontsize=20)




def main():
	
	return 0

if __name__ == '__main__':
	main()

