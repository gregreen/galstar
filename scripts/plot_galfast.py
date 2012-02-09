#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       plot_galfast.py
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
import numpy as np
import pyfits
import argparse, sys
from scipy import interpolate
from os.path import abspath
import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


A_ratio = [5.155/2.751, 3.793/2.751, 1., 2.086/2.751, 1.479/2.751]
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
	ra_dec = np.empty((N,2), dtype=float)	# (RA, DEC)
	mags = np.empty((N,5), dtype=float)		# (u, g, r, i, z)
	errs = np.empty((N,5), dtype=float)		# (sigma_u, ..., sigma_z)
	params = np.empty((N,4), dtype=float)	# (DM, Ar, Mr, FeH)
	absmags = np.empty((N,5), dtype=float)	# (M_u, ..., M_z)
	for i,d in enumerate(data):
		ra_dec[i] = d[1]
		mags[i] = d[11][:-1]	# Observed ugriz
		A_r, DM = d[7], d[3]
		for j,m in enumerate(mags[i]):
			errs[i,j] = interpolate.splev(m, err_spline[j])
			absmags[i,j] = mags[i,j] - DM - A_r*A_ratio[j]
		params[i,0], params[i,1], params[i,2], params[i,3] = d[3], d[7], d[4], d[6]
	# Return [(RA, DEC),...] , [(u,g,r,i,z),...], [(sig_u,sig_g,...),...], [(DM,Ar,Mr,FeH),...]
	return ra_dec, mags, errs, params, absmags


# Plot g-r vs. M_r to given axes
def plot_HR(absmags, ax):
	ax.plot(absmags[:,1]-absmags[:,2], absmags[:,2], '.', linestyle='None', markersize=0.5)
	ax.set_xlabel(r'$g - r$', fontsize=16)
	ax.set_ylabel(r'$M_r$', fontsize=16)
	ylim = ax.get_ylim()
	ax.set_ylim(ylim[1],ylim[0])


# Plot DM vs. Ar
def plot_DM_Ar(params, ax):
	ax.plot(params[:,0], params[:,1], '.', linestyle='None', markersize=0.5)
	ax.set_xlabel(r'$\mu$', fontsize=16)
	ax.set_ylabel(r'$A_r$', fontsize=16)


# Plot Mr vs. Z
def plot_Mr_Z(params, absmags, ax):
	ax.plot(absmags[:,2], params[:,3], '.', linestyle='None', markersize=0.5)
	ax.set_xlabel(r'$M_r$', fontsize=16)
	ax.set_ylabel(r'$Z$', fontsize=16)
	xlim = ax.get_xlim()
	ax.set_xlim(xlim[1],xlim[0])


# Plot DM vs. Mr
def plot_DM_Mr(params, ax):
	ax.plot(params[:,0], params[:,2], '.', linestyle='None', markersize=0.5)
	ax.set_xlabel(r'$\mu$', fontsize=16)
	ax.set_ylabel(r'$M_r$', fontsize=16)
	ylim = ax.get_ylim()
	ax.set_ylim(ylim[1],ylim[0])


def main():
	parser = argparse.ArgumentParser(prog='plot_galfast',  description='Plot various aspects of galfast output', add_help=True)
	parser.add_argument('filename', type=str, help='Galfast FITS file')
	parser.add_argument('--filter', dest='filt', type=float, default=None, help='Filter objects with errors greater than the specified amount')
	parser.add_argument('--output', type=str, help='Output filename for plot')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Get stellar data
	print 'Loading data...'
	init_errs()
	ra_dec, mags, errs, params, absmags = get_objects(values.filename, filt=values.filt)
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	# Set up figure
	fig = plt.figure(figsize=(8.5,11))
	fig.suptitle(r'$\mathrm{Galfast\ Output}$', fontsize=24, y=0.95)
	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)
	
	fig.subplots_adjust(wspace=0.25, hspace=0.22, top=0.88, right=0.91, left=0.12)
	
	# Plot Herzsprung-Russel Diagram
	print 'Generating H-R diagram...'
	plot_HR(absmags, ax1)
	
	# Plot distance modulus vs. reddening
	print 'Plotting DM vs. A_r...'
	plot_DM_Ar(params, ax2)
	
	# Plot red magnitude vs. metallicity
	print 'Plotting M_r vs Z...'
	plot_Mr_Z(params, absmags, ax3)
	
	# Plot red magnitude vs. metallicity
	print 'Plotting DM vs M_r...'
	plot_DM_Mr(params, ax4)
	
	# Save plot to file
	if values.output != None:
		fn = abspath(values.output)
		if '.' not in fn:
			fn += '.png'
		fig.savefig(fn, dpi=300)
	
	plt.show()
	
	print 'Done.'
	
	return 0

if __name__ == '__main__':
	main()

