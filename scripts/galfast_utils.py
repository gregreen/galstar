#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       galfast_utils.py
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


import numpy as np
import pyfits
from scipy import interpolate
from struct import unpack
from math import sqrt


DM, Ar, Mr, FeH = range(4)
A_ratio = np.array([1.8236, 1.4241, 1.0000, 0.7409, 0.5821])


# Read in a statistics file from a galstar run
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

# Load the photometric errors in each band as a function of magnitude
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


# Load information on the objects in a galfast FITS file
def get_objects(galfast_fits_fname, filt=0.5):
	if err_spline == None:
		init_errs()
	data = pyfits.getdata(galfast_fits_fname, ext=1)
	N = len(data)
	ra_dec = np.empty((N,2), dtype=float)
	obs_mags = np.empty((N,5), dtype=float)
	LSST_mags = np.empty((N,5), dtype=float)
	abs_mags = np.empty((N,5), dtype=float)
	errs = np.empty((N,5), dtype=float)
	params = np.empty((N,4), dtype=float)	# (DM, Ar, Mr, FeH)
	for i,d in enumerate(data):
		ra_dec[i] = d[1]
		obs_mags[i] = d[11][:-1]	# Observed ugriz (with photometric errors)
		LSST_mags[i] = d[9][:-1]	# "Real" apparent ugriz (without photometric errors)
		for j,m in enumerate(obs_mags[i]):
			errs[i,j] = interpolate.splev(m, err_spline[j])
		params[i,DM], params[i,Ar], params[i,Mr], params[i,FeH] = d[3], d[7], d[4], d[6]
		abs_mags[i] = LSST_mags[i] - A_ratio*params[i,Ar]	# Absolute magnitudes in ugriz (without photometric errors)
	# Return [(RA, DEC),...] , [(u,g,r,i,z),...], [(sig_u,sig_g,...),...], [(DM,Ar,Mr,FeH),...]
	return ra_dec, obs_mags, LSST_mags, abs_mags, errs, params


# Spline interpolation of a catalog of SEDs, loaded from an ascii file
class SED:
	# Initialize the SED catalog, using an ascii file
	def __init__(self, fname='../data/MSandRGBcolors_v1.3.dat'):
		# Load raw data
		Mr, FeH, ug, gr, ri, iz, zy = np.loadtxt(fname, usecols=(0,1,2,3,4,5,6), unpack=True)
		Mg = Mr + gr
		Mu = Mg + ug
		Mi = Mr - ri
		Mz = Mi - iz
		My = Mz - zy
		
		# Set up interpolating functions
		FeH_range = np.arange(-2.50, 0.05, 0.05)
		Mr_range = np.arange(-1.00, 28.01, 0.01)
		Mu = np.array(Mu).reshape(len(FeH_range), len(Mr_range)).T
		Mg = np.array(Mg).reshape(len(FeH_range), len(Mr_range)).T
		Mi = np.array(Mi).reshape(len(FeH_range), len(Mr_range)).T
		Mz = np.array(Mz).reshape(len(FeH_range), len(Mr_range)).T
		My = np.array(My).reshape(len(FeH_range), len(Mr_range)).T
		
		self.u = interpolate.RectBivariateSpline(Mr_range, FeH_range, Mu)
		self.g = interpolate.RectBivariateSpline(Mr_range, FeH_range, Mg)
		self.i = interpolate.RectBivariateSpline(Mr_range, FeH_range, Mi)
		self.z = interpolate.RectBivariateSpline(Mr_range, FeH_range, Mz)
		self.y = interpolate.RectBivariateSpline(Mr_range, FeH_range, My)
	
	# Return (M_u, M_g, M_r, M_i, M_z, M_y) for a given (M_r, [Fe/H]), using spline interpolation of the catalog values
	def SED(self, Mr, FeH):
		return np.array([self.u(Mr,FeH)[0][0], self.g(Mr,FeH)[0][0], Mr, self.i(Mr,FeH)[0][0], self.z(Mr,FeH)[0][0], self.y(Mr,FeH)[0][0]])
	
	# Allow the class object to be called as a function
	def __call__(self, Mr, FeH):
		return self.SED(Mr, FeH)

