#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       plotpdf.py
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

import sys, argparse
from os.path import abspath

from galstar_io import *

import numpy as np

import matplotlib as mplib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# Plot the given probability surfaces (p) to a single figure, saving if given a filename
def plot_surfs(p, bounds, clip=None, shape=(3,2), fname=None, labels=None, converged=None):
	# Set up the figure
	fig = plt.figure(figsize=(8.5,11.), dpi=100)
	grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.2, share_all=True, aspect=False)
	
	# Show the probability surfaces
	N_plots = min(p.shape[0], shape[0]*shape[1])
	for i in xrange(N_plots):
		grid[i].imshow(p[i].T, extent=bounds, origin='lower', aspect='auto', cmap='hot', interpolation='nearest')
	
	# Clip domain and range
	if clip != None:
		grid[0].set_xlim(clip[0:2])
		grid[0].set_ylim(clip[2:4])
	
	# Label axes
	if labels != None:
		for ax in grid.axes_row[-1]:
			ax.set_xlabel(r'$%s$' % labels[0], fontsize=16)
		for ax in grid.axes_column[0]:
			ax.set_ylabel(r'$%s$' % labels[1], fontsize=16)
	
	# Mark whether each star converged
	if converged != None:
		for i in xrange(N_plots):
			if not converged[i]:
				xmax, ymax = grid[i].get_xlim()[1], grid[i].get_ylim()[1]
				x, y = 0.95*xmax, 0.95*ymax
				grid[i].text(x, y, '!', color='white', fontsize=24, horizontalalignment='right', verticalalignment='top')
				#grid[i].scatter(x, y, color='r', s=10)
	
	if fname != None:
		print 'Saving figure to %s ...' % fname
		fig.savefig(abspath(fname), dpi=150)
	
	return fig


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='plotpdf', description='Plots posterior distributions produced by galstar', add_help=True)
	parser.add_argument('binfn', type=str, help='File containing binned probability density functions for each star along l.o.s. (also accepts gzipped files)')
	parser.add_argument('statsfn', type=str, help='File containing summary statistics for each star.')
	parser.add_argument('plotfn', type=str, help='Base filename (without extension) for plots.')
	parser.add_argument('-se', '--startend', type=int, nargs=2, default=(0,6), help='Start and end indices (default: 0 6).')
	parser.add_argument('-rc', '--rowcol', type=int, nargs=2, default=(3,2), help='# of rows and columns, respectively (default: 2 3)')
	parser.add_argument('-sh', '--show', action='store_true', help='Show plot of result.')
	parser.add_argument('-sm', '--smooth', type=int, nargs=2, default=(1,1), help='Std. dev. of smoothing kernel (in pixels) for individual pdfs (default: 1 1).')
	parser.add_argument('-y', '--ymax', type=float, default=None, help='Upper bound on y in plots')
	parser.add_argument('-p', '--params', type=str, nargs=2, default=('DM','Ar'), help='Name of x- and y-axes, respectively (default: DM Ar). Choices are (DM, Ar, Mr, FeH).')
	#parser.add_argument('-cv', '--converged', action='store_true', help='Show only converged stars')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Determine axis labels
	param_dict = {'dm':'\mu', 'ar':'A_r', 'mr':'M_r', 'feh':'Z'}
	labels = []
	for p in values.params:
		try:
			labels.append(param_dict[p.lower()])
		except:
			print 'Invalid parameter name: "%s"' % p
			print 'Valid parameter names are DM, Ar, Mr and FeH.'
			return 1
	
	# Determine number of figures to make
	if values.startend[1] <= values.startend[0]:
		print 'Invalid input for --startend: "%d %d". The ending index must be greater than the starting index.' % values.startend
	N_stars = values.startend[1] - values.startend[0]
	N_ax = values.rowcol[0] * values.rowcol[1]
	N_fig = int(np.ceil(float(N_stars) / float(N_ax)))
	
	np.seterr(all='ignore')
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# For each figure, load in only the necessary probability density functions, and pass each to the plotter
	for i in xrange(N_fig):
		print 'Plotting figure %d of %d ...' % (i+1, N_fig)
		plotfn = None
		if N_fig != 1:
			plotfn = '%s_%d.png' % (values.plotfn, i)
		else:
			plotfn = '%s.png' % (values.plotfn)
		selection = values.startend[0] + np.arange(i*N_ax, min((i+1)*N_ax, N_stars))
		bounds, p = load_bins(values.binfn, selection)
		clip = None
		if values.ymax != None:
			clip = list(bounds)
			clip[3] = values.ymax
		p = smooth_bins(p, values.smooth)
		converged, mean, cov = load_stats(values.statsfn, selection)
		#p[p == 0] = np.min(p[p != 0])
		#p = np.log(p)
		#p[np.logical_not(np.isfinite(p))] = np.min(p[np.isfinite(p)])
		fig = plot_surfs(p, bounds, clip, values.rowcol, plotfn, labels, converged)
		if not values.show:
			plt.close(fig)
	
	if values.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

