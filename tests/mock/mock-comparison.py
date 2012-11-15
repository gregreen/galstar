#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  mock-comparison.py
#  
#  Copyright 2012 Greg Green <greg@greg-UX31A>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplib

from scipy.interpolate import interp2d, RectBivariateSpline
import scipy.ndimage.interpolation as interp
import scipy.stats

import argparse, sys
from os.path import abspath

from galstar_io import load_bins, load_stacked_bins_sparse, load_stats, smooth_bins


def stack_shifted(bounds, p, shift, norm):
	dx = shift[0] * p.shape[1] / (bounds[1] - bounds[0])
	dy = shift[1] * p.shape[2] / (bounds[3] - bounds[2])
	dxy = np.vstack([dx,dy]).T
	p_stacked = np.zeros(p.shape[1:], dtype='f8')
	for surf,D,Z in zip(p,dxy,norm):
		tmp = interp.shift(surf, D) / Z
		p_stacked += tmp #*= tmp + 1.e-5*np.exp(-tmp/1.e-2)
	return p_stacked


def P_star(bounds, p, truth):
	idx_DM = ( (truth['DM'] - bounds[0]) / (bounds[1] - bounds[0])
	                                         * p.shape[1] ).astype('i8')
	idx_Ar = ( (truth['Ar'] - bounds[2]) / (bounds[3] - bounds[2])
	                                         * p.shape[2] ).astype('i8')
	
	idx = [np.arange(p.shape[0]), idx_DM, idx_Ar]
	
	threshold = p[idx]
	
	P_ret = np.empty(p.shape[0], dtype='f8')
	for i,pp in enumerate(p):
		idx = pp > threshold[i]
		gtr, less = np.sum(pp[idx]), np.sum(pp[~idx])
		P_ret[i] = less / (gtr + less)
	
	return P_ret


def binom_confidence(nbins, ntrials, confidence):
	q = 0.5 * (1. - confidence)
	
	qlower = 1. - q**(1./nbins)
	qupper = q**(1./nbins)
	
	rv = scipy.stats.binom(ntrials, float(nbins)/float(ntrials))
	
	P = rv.cdf(np.arange(nbins+1))
	lower = np.where(P >= qlower)[0][0]
	upper = np.where(P < qupper)[0][-1] + 1
	
	return lower, upper


def main():
	parser = argparse.ArgumentParser(
	              prog='mock-comparison.py',
	              description='Compares results from galstar for mock data '
	                          'with true stellar parameters.',
	              add_help=True)
	parser.add_argument('bin', type=str, help='Binned pdfs from galstar')
	parser.add_argument('stats', type=str, help='Statistics produced by galstar')
	parser.add_argument('truth', type=str, help='Test input used by galstar')
	parser.add_argument('--stack-out', '-so', type=str, default=None,
	                       help='Output filename for stacked pdf plot.')
	parser.add_argument('--pct-out', '-po', type=str, default=None,
	                        help='Output filename for percentile plot.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	if (values.stack_out == None) and (values.pct_out == None):
		print "Either '--stack-out' or '--pct-out' (or both) must be specified."
		return 0
	
	test_fname = abspath(values.truth)
	dtype = [('DM', 'f8'), ('Ar', 'f8'), ('Mr', 'f8'), ('FeH', 'f8')]
	truth = np.loadtxt(test_fname, dtype=dtype, skiprows=7)
	
	stats_fname = abspath(values.stats)
	bin_fname = abspath(values.bin)
	converged, ln_evidence, mean, cov = load_stats('stats.dat')
	bounds, p = load_bins('DM_Ar.dat')
	p = smooth_bins(p, [2,2])
	norm = np.sum(np.sum(p, axis=1), axis=1)
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	# Shifted and stacked pdfs
	if values.stack_out != None:
		stack_fname = abspath(values.stack_out)
		
		# Simple statistics
		Delta_DM = (truth['DM']-mean[:,0]) / np.sqrt(cov[:,0,0])
		Delta_Ar = (truth['Ar']-mean[:,1]) / np.sqrt(cov[:,1,1])
		Delta_Mr = (truth['Mr']-mean[:,2]) / np.sqrt(cov[:,2,2])
		Delta_FeH = (truth['FeH']-mean[:,3]) / np.sqrt(cov[:,3,3])
		
		w_x = bounds[1] - bounds[0]
		w_y = bounds[3] - bounds[2]
		dx = bounds[0] + 0.5*w_x - truth['DM']
		dy = bounds[2] + 0.5*w_y - truth['Ar']
		bounds_new = [-0.5*w_x, 0.5*w_x, -0.5*w_y, 0.5*w_y]
		stack = stack_shifted(bounds, p, [dx,dy], norm)
		
		DM_range = np.linspace(bounds_new[0], bounds_new[1], stack.shape[0])
		p_DM = np.sum(stack, axis=1)
		p_DM /= np.sum(p_DM)
		
		Ar_range = np.linspace(bounds_new[2], bounds_new[3], stack.shape[1])
		p_Ar = np.sum(stack, axis=0)
		p_Ar /= np.sum(p_Ar)
		
		# Determine geometry of density plot and histograms
		main_left, main_bottom = 0.18, 0.16
		main_width, main_height = 0.63, 0.65
		buffer_right, buffer_top = 0., 0.
		histx_height, histy_width = 0.12, 0.09
		rect_main = [main_left, main_bottom, main_width, main_height]
		rect_histx = [main_left, main_bottom+main_height+buffer_top, main_width, histx_height]
		rect_histy = [main_left+main_width+buffer_right, main_bottom, histy_width, main_height]
		
		# Set up the figure with a density plot and two histograms
		fig = plt.figure(figsize=(4,3), dpi=150)
		ax_density = fig.add_axes(rect_main)
		ax_histx = fig.add_axes(rect_histx)
		ax_histy = fig.add_axes(rect_histy)
		
		xlim = [-2.,2.]
		ylim = [-1.,1.]
		
		ax_density.imshow(stack.T, extent=bounds_new, origin='lower', vmin=0.,
						  aspect='auto', cmap='hot', interpolation='nearest')
		ax_density.plot([0., 0.], [ylim[0]-1.,ylim[1]+1.], 'c:', lw=0.5, alpha=0.35)
		ax_density.plot([xlim[0]-1., xlim[1]+1.], [0., 0.], 'c:', lw=0.5, alpha=0.35)
		ax_density.set_xlim(xlim)
		ax_density.set_ylim(ylim)
		
		ax_histx.fill_between(DM_range, y1=p_DM, alpha=0.4, facecolor='b')
		ax_histx.plot([0., 0.], [0., 1.1*np.max(p_DM)], 'g-', lw=0.5)
		ax_histx.set_ylim(0., 1.1*np.max(p_DM))
		ax_histx.set_xlim(xlim)
		ax_histx.set_xticklabels([])
		ax_histx.set_yticklabels([])
		
		ax_histy.fill_betweenx(Ar_range, x1=p_Ar, alpha=0.4, facecolor='b')
		ax_histy.plot([0., 1.1*np.max(p_Ar)], [0., 0.], 'g-', lw=0.5)
		ax_histy.set_xlim(0., 1.1*np.max(p_Ar))
		ax_histy.set_ylim(ylim)
		ax_histy.set_xticklabels([])
		ax_histy.set_yticklabels([])
		
		fig.savefig(stack_fname, dpi=300)
	
	# Percentile statistics
	if values.pct_out != None:
		pct_fname = abspath(values.pct_out)
		
		P_indiv = P_star(bounds, p, truth)
		
		
		fig = plt.figure(figsize=(4,3), dpi=200)
		ax = fig.add_subplot(1,1,1)
		ax.hist(P_indiv)
		ax.set_xlim(0., 1.)
		ax.set_xlabel(r'$\% \mathrm{ile}$', fontsize=16)
		ax.set_ylabel(r'$\mathrm{\# \ of \ stars}$', fontsize=16)
		fig.subplots_adjust(left=0.18, bottom=0.18)
		
		fig.savefig(pct_fname, dpi=300)
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

