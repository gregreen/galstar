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
def plot_surfs(p, bounds, clip=None, shape=(3,2), figsize=(8.5, 11.), fname=None, labels=None, converged=None, ln_evidence=None, true_params=None):
	# Set up the figure
	fig = plt.figure(figsize=figsize, dpi=300)
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
				x, y = 0.90*xmax, 0.95*ymax
				grid[i].text(x, y, '!', color='white', fontsize=24, horizontalalignment='right', verticalalignment='top')
				#grid[i].scatter(x, y, color='r', s=10)
	
	# Write evidence for each star
	if ln_evidence != None:
		for i in xrange(N_plots):
			xmax, ymax = grid[i].get_xlim()[1], grid[i].get_ylim()[1]
			x, y = 0.95*xmax, 0.95*ymax
			grid[i].text(x, y, '%.2g' % ln_evidence[i], color='white', fontsize=14, horizontalalignment='right', verticalalignment='top')
	
	# Mark true stellar parameters
	if true_params != None:
		for i in xrange(N_plots):
			grid[i].scatter(true_params[0][i], true_params[1][i], c='g', s=25)
	
	fig.subplots_adjust(bottom=0.15)
	
	if fname != None:
		print 'Saving figure to %s ...' % fname
		fig.savefig(abspath(fname), dpi=300)
	
	return fig


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='plotpdf', description='Plots posterior distributions produced by galstar', add_help=True)
	parser.add_argument('--binfn', nargs='+', type=str, required=True, help='File containing binned probability density functions for each star along l.o.s. (also accepts gzipped files)')
	parser.add_argument('--statsfn', nargs='+', type=str, required=True, help='File containing summary statistics for each star.')
	parser.add_argument('-o', '--plotfn', type=str, required=True, help='Base filename (without extension) for plots.')
	parser.add_argument('--testfn', nargs='+', type=str, default=None, help='ASCII file with true stellar parameters (same as used for galstar input).')
	parser.add_argument('-se', '--startend', type=int, nargs=2, default=(0,6), help='Start and end indices (default: 0 6).')
	parser.add_argument('-rc', '--rowcol', type=int, nargs=2, default=(3,2), help='# of rows and columns, respectively (default: 2 3)')
	parser.add_argument('-sh', '--show', action='store_true', help='Show plot of result.')
	parser.add_argument('-sm', '--smooth', type=int, nargs=2, default=(1,1), help='Std. dev. of smoothing kernel (in pixels) for individual pdfs (default: 1 1).')
	parser.add_argument('-y', '--ymax', type=float, default=None, help='Upper bound on y in plots.')
	parser.add_argument('-p', '--params', type=str, nargs=2, default=('DM','Ar'), help='Name of x- and y-axes, respectively (default: DM Ar). Choices are (DM, Ar, Mr, FeH).')
	parser.add_argument('-cnv', '--converged', action='store_true', help='Show only converged stars.')
	parser.add_argument('-ev', '--evidence', type=float, default=None, help='Filter out stars which have evidence X below max. evidence.')
	parser.add_argument('-stk', '--stack', action='store_true', help='Stack stellar pdfs.')
	parser.add_argument('--nomarks', action='store_true', help='Do not show evidence or convergence flag.')
	parser.add_argument('-fig', '--figsize', type=float, nargs=2, default=(8.5, 11.), help='Figure width and height in inches.')
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
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# Load in pdfs
	pdf = []
	converged = []
	ln_evidence = []
	param_true = [[],[]]
	bounds = None
	N_stars = 0
	for i,(sf,bf) in enumerate(zip(values.statsfn, values.binfn)):
		conv, ln_ev, mean, cov = load_stats(sf)
		bounds, p = load_bins(bf, True)
		converged.append(conv)
		pdf.append(p)
		ln_evidence.append(ln_ev)
		if values.testfn != None:
			tmp_param = [None, None, None, None]
			lb, tmp_param[0], tmp_param[1], tmp_param[2], tmp_param[3] = load_true(values.testfn[i])
			param_list = ['dm', 'ar', 'mr', 'feh']
			for i in range(2):
				for k,l in enumerate(param_list):
					if values.params[i].lower() == l:
						param_true[i].append(tmp_param[k])
						break
		N_stars += conv.size
		if N_stars >= values.startend[1]:
			break
	pdf = np.vstack(pdf)
	converged = np.hstack(converged)
	ln_evidence = np.hstack(ln_evidence)
	if values.testfn != None:
		param_true[0] = np.hstack(param_true[0])
		param_true[1] = np.hstack(param_true[1])
	
	pdf[~np.isfinite(pdf)] = 0.
	idx = (np.sum(np.sum(pdf, axis=1), axis=1) != 0.)
	if values.converged:
		idx = idx & converged
	if values.evidence != None:
		idx = idx & (ln_evidence >= np.max(ln_evidence) - values.evidence)
	#print idx
	#print np.sum(np.sum(pdf, axis=1), axis=1)[0]
	#print pdf[0][pdf[0] > 0.]
	
	pdf = pdf[idx]
	ln_evidence = ln_evidence[idx]
	if values.testfn != None:
		param_true[0] = param_true[0][idx]
		param_true[1] = param_true[1][idx]
	converged = converged[idx]
	N_stars = pdf.shape[0]
	pdf = smooth_bins(pdf, values.smooth)
	
	#print np.sum(np.sum(pdf, axis=1), axis=1)[0]
	#print pdf[0][pdf[0] > 0.]
	#print param_true
	
	clip = None
	if values.ymax != None:
		clip = list(bounds)
		clip[3] = values.ymax
	
	# Generate figures
	if values.stack:
		print 'Plotting stacked pdfs ...'
		plotfn = '%s.png' % (values.plotfn)
		imin, imax = values.startend[0], np.min([N_stars, values.startend[1]])
		#print pdf.shape
		pdf = np.sum(pdf[imin:imax], axis=0)
		w,h = pdf.shape
		pdf.shape = (1, w, h)
		fig = plot_surfs(pdf, bounds, clip, [1,1], values.figsize, plotfn, labels)
	else:
		for i in range(values.startend[0],np.min([N_stars, values.startend[1]]),
		                                        np.prod(values.rowcol)):
			imax = np.min([i+6, converged.size, values.startend[1]])
			print 'Plotting pdfs %d through %d of %d ...' % (i+1, imax, N_stars)
			
			plotfn = None
			if np.min([N_stars, values.startend[1]]) - values.startend[0] <= np.prod(values.rowcol):
				plotfn = '%s.png' % (values.plotfn)
			else:
				plotfn = '%s_%d.png' % (values.plotfn,
				                       (i-values.startend[0])/np.prod(values.rowcol))
			
			true_params = None
			if values.testfn != None:
				true_params = (param_true[0][i:imax], param_true[1][i:imax])
			
			if values.nomarks:
				fig = plot_surfs(pdf[i:imax], bounds, clip,
				                 values.rowcol, values.figsize,
				                 plotfn, labels, true_params=true_params)
			else:
				fig = plot_surfs(pdf[i:imax], bounds, clip,
				                 values.rowcol, values.figsize,
				                 plotfn, labels, converged[i:imax],
				                 ln_evidence[i:imax], true_params)
			
			if not values.show:
				plt.close(fig)
	
	if values.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

