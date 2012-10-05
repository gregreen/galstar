#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gc_comp.py
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

import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
import pyfits

from galstar_io import *
from plot_priors import TGalacticModel


def main():
	galstar_output_dir = 'data' #'/media/FreeAgent GoFlex Drive/Data'
	gcdata = pyfits.getdata('mwgc.fits')
	
	MessierGC = ['NGC 7089',
	             'NGC 5272',
	             'NGC 6121',
	             'NGC 5904',
	             'NGC 6333',
	             'NGC 6254',
	             'NGC 6218',
	             'NGC 6205',
	             'NGC 6402',
	             'NGC 7078',
	             'NGC 6273',
	             'NGC 6656',
	             'NGC 6626',
	             'NGC 7099',
	             'NGC 5024',
	             'NGC 6715',
	             'NGC 6809',
	             'NGC 6779',
	             'NGC 6266',
	             'NGC 4590',
	             'NGC 6637',
	             'NGC 6681',
	             'NGC 6838',
	             'NGC 6981',
	             'NGC 6864',
	             'NGC 1904',
	             'NGC 6093',
	             'NGC 6341',
	             'NGC 6171']
	index = [2, 5, 6, 8, 9, 13, 19, 23, 25, 27, 28]
	gcID_list = []
	for i in index:
		gcID_list.append(MessierGC[i])
	
	model = TGalacticModel()
	
	# Set matplotlib style attributes
	mplib.rc('text', usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('ytick.major', size=6)
	mplib.rc('ytick.minor', size=4)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('axes', grid=False)
	
	Delta_DM = np.empty(len(gcID_list), dtype='f4')
	Delta_Ar = np.empty(len(gcID_list), dtype='f4')
	
	p0 = 1.e-5
	
	print '# Comparison of published globular cluster distances and'
	print "# extinctions with galstar's max-likelihood estimates."
	print '# Stellar probability densities p(DM, Ar) are softened with'
	print '# p_0 = 1.e-5.'
	print '#'
	print '#  ID     DM_ML   Ar_ML  DM_GC   Ar_GC'
	#print 'NGC_1234  10.000  1.000  11.000  1.100'
	
	for i,gcID in enumerate(gcID_list):
		#print 'Plotting %s ...' % gcID
		
		# Read in GC properties from catalog
		gc = gcdata[gcdata['ID'] == gcID]
		DM_gc = 5. * (2. + np.log10(gc['R_Sun'][0]))
		Ar_gc = 2.271 * gc['EBV']
		
		# Load in stacked stellar probability surfaces
		stats_fname = '%s/%d.stats' % (galstar_output_dir, i)
		surface_fname = '%s/%d_DM_Ar.dat' % (galstar_output_dir, i)
		
		conv, ln_ev, mean, cov = load_stats(stats_fname)
		idx = conv & (ln_ev >= np.max(ln_ev) - 30.)
		sel = np.where(idx)[0]
		
		bounds, p = load_stacked_bins_sparse(surface_fname,
		                                     p0=p0,
		                                     selection=sel,
		                                     verbose=False)
		
		# Define bin centers
		dDM = (bounds[1] - bounds[0]) / float(p.shape[0])
		DM_range = np.linspace(bounds[0]+dDM/2., bounds[1]-dDM/2., p.shape[0])
		dAr = (bounds[3] - bounds[2]) / float(p.shape[1])
		Ar_range = np.linspace(bounds[2]+dAr/2., bounds[3]-dAr/2., p.shape[1])
		
		# Find max-likelihood point
		j,k = np.unravel_index(np.argmax(p), p.shape)
		DM_ML = DM_range[j]
		Ar_ML = Ar_range[k]
		Delta_DM[i] = DM_ML - DM_gc
		Delta_Ar[i] = Ar_ML - Ar_gc
		
		print '%s  %.3f  %.3f  %.3f  %.3f' % (gcID.replace(' ', '_'),
		                                      DM_ML, Ar_ML,
		                                      DM_gc, Ar_gc)
		
		# Project probability onto DM and Ar separately
		if p0 > 0.:
			p_linear = np.exp(p - np.max(p))
			p_DM = np.sum(p_linear, axis=1)
			p_Ar = np.sum(p_linear, axis=0)
		else:
			p_DM = np.sum(p, axis=1)
			p_Ar = np.sum(p, axis=0)
		
		max_Ar_pixel = None
		max_DM_pixel = None
		min_DM_pixel = None
		if p0 > 0.:
			tmp = np.sum(p - np.max(p), axis=0)
			max_Ar_pixel = np.max(np.where(tmp > np.min(tmp) + 0.01*(np.max(tmp)-np.min(tmp))))
			tmp = np.sum(p - np.max(p), axis=1)
			max_DM_pixel = np.max(np.where(tmp > np.min(tmp) + 0.01*(np.max(tmp)-np.min(tmp))))
			min_DM_pixel = np.min(np.where(tmp > np.min(tmp) + 0.01*(np.max(tmp)-np.min(tmp))))
		else:
			max_Ar_pixel = np.max(np.where(p_Ar > 1.e-2*np.max(p_Ar)))
			max_DM_pixel = np.max(np.where(p_DM > 1.e-2*np.max(p_DM)))
			min_DM_pixel = np.min(np.where(p_DM > 1.e-2*np.max(p_DM)))
		max_Ar = (max_Ar_pixel / float(p_Ar.size) * (bounds[3]-bounds[2])
		                                                    + bounds[2])
		y_max = min([max([max_Ar, 1.2 * Ar_gc]), bounds[3]])
		max_DM = (max_DM_pixel / float(p_DM.size) * (bounds[1]-bounds[0])
		                                                    + bounds[0])
		x_max = min([max([max_DM, 1.2 * DM_gc]), bounds[1]])
		min_DM = (min_DM_pixel / float(p_DM.size) * (bounds[1]-bounds[0])
		                                                    + bounds[0])
		x_min = max([min([min_DM, 0.8 * DM_gc]), bounds[0]])
		
		# Convert projected probabilities from log to linear units
		#if p0 > 0.:
		#	p_DM = np.exp(p_DM - np.max(p_DM))
		#	p_Ar = np.exp(p_Ar - np.max(p_Ar))
		
		# Determine geometry of density plot and histograms
		main_left, main_bottom = 0.18, 0.16
		main_width, main_height = 0.63, 0.65
		buffer_right, buffer_top = 0., 0.
		histx_height, histy_width = 0.12, 0.09
		rect_main = [main_left, main_bottom, main_width, main_height]
		rect_histx = [main_left, main_bottom+main_height+buffer_top, main_width, histx_height]
		rect_histy = [main_left+main_width+buffer_right, main_bottom, histy_width, main_height]
		
		# Set up the figure with a density plot and two histograms
		fig = plt.figure(figsize=(4,3), dpi=200)
		ax_density = fig.add_axes(rect_main)
		ax_histx = fig.add_axes(rect_histx)
		ax_histy = fig.add_axes(rect_histy)
		
		# Plot stacked surfaces with catalog positions of GCs overlaid
		ax_density.imshow(p.T, extent=bounds, origin='lower',
		                  aspect='auto', cmap='hot', interpolation='nearest')
		ax_density.scatter(DM_gc, Ar_gc, s=20, c='g', marker='+')
		ax_density.set_xlabel(r'$\mu$', fontsize=16)
		ax_density.set_ylabel(r'$A_{r}$', fontsize=16)
		ax_density.set_ylim(bounds[2], y_max)
		#ax_density.set_xlim(bounds[0], bounds[1])
		ax_density.set_xlim(x_min, x_max)
		
		# DM histogram
		norm = np.sum(p_DM[1:-1]) + 0.5 * (p_DM[0] + p_DM[-1])
		p_DM /= norm
		ax_histx.fill_between(DM_range, y1=p_DM, alpha=0.4, facecolor='b')
		
		# Expected number of stars in each distance bin
		l, b = gc['l']*np.pi/180., gc['b']*np.pi/180.
		cos_l, sin_l = np.cos(l), np.sin(l)
		cos_b, sin_b = np.cos(b), np.sin(b)
		dNdDM = np.array([model.dn_dDM(x, cos_l, sin_l, cos_b, sin_b)[0] * model.dn_dDM_corr(x, m_max=23.) for x in DM_range])
		norm = None
		if p0 > 0.:
			norm = np.max(dNdDM) / np.max(p_DM)
		else:
			norm = np.sum(dNdDM[1:-1]) + 0.5 * (dNdDM[0] + dNdDM[-1])
		dNdDM /= norm
		ax_histx.fill_between(DM_range, y1=dNdDM, alpha=0.4, facecolor='r')
		
		ylim = list(ax_histx.get_ylim())
		ax_histx.plot([DM_gc, DM_gc], ylim, 'g-')
		
		# Set DM histogram limits
		ax_histx.set_ylim(0, ylim[1])
		ax_histx.set_xlim(x_min, x_max)
		ax_histx.set_xticklabels([])
		ax_histx.set_yticklabels([])
		
		# Ar histogram
		ax_histy.fill_betweenx(Ar_range, x1=p_Ar, alpha=0.4, facecolor='b')
		xlim = list(ax_histy.get_xlim())
		ax_histy.plot(xlim, [Ar_gc, Ar_gc], 'g-')
		ax_histy.set_xlim(0, xlim[1])
		ax_histy.set_ylim(bounds[2], y_max)
		ax_histy.set_xticklabels([])
		ax_histy.set_yticklabels([])
		
		fig.savefig('%s.png' % gcID, dpi=200)
		#plt.show()
	
	#print Delta_DM
	#print Delta_Ar
	
	print ''
	print '# Summary statistics:'
	print '#'
	print '# Delta_DM = %.3f' % np.mean(Delta_DM)
	print '# sigma_DM = %.3f' % np.std(Delta_DM)
	print '#'
	print '# Delta_Ar = %.3f' % np.mean(Delta_Ar)
	print '# sigma_Ar = %.3f' % np.std(Delta_Ar)
	print '#'
	
	
	return 0

if __name__ == '__main__':
	main()

