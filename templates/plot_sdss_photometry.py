#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#  untitled.py
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
import pyfits
import numpy as np
import scipy.interpolate
import cPickle as pickle

import os

import iterators


def density_scatter(ax, x, y, nbins=(50,50), threshold=5):
	'''
	Draw a combination density map / scatterplot to the given axes.
	
	Adapted from answer to stackoverflow question #10439961
	'''
	# Make histogram of data
	bounds = [[np.min(x)-1.e-10, np.max(x)+1.e-10], [np.min(y)-1.e-10, np.max(y)+1.e-10]]
	h, loc_x, loc_y = scipy.histogram2d(x, y, range=bounds, bins=nbins)
	pos_x, pos_y = np.digitize(x, loc_x), np.digitize(y, loc_y)
	
	# Mask histogram points below threshold
	idx = (h[pos_x - 1, pos_y - 1] < threshold)
	h[h < threshold] = np.nan
	
	# Density plot
	img = ax.imshow(np.log(h.T), origin='lower', cmap='jet', extent=np.array(bounds).flatten(), interpolation='none', aspect='auto')
	
	# Scatterplot
	ax.scatter(x[idx], y[idx], c='b', s=1)
	
	return img

def gr_vs_logg(d):
	idx = (d['PSFMAG_u'] > 0.) & (d['PSFMAG_g'] > 0.) & (d['PSFMAG_r'] > 0.) & (d['PSFMAG_i'] > 0.) & (d['PSFMAG_z'] > 0.)
	idx = idx & (d['PSFMAGERR_u'] < 0.1) & (d['PSFMAGERR_g'] < 0.1) & (d['PSFMAGERR_r'] < 0.1) & (d['PSFMAGERR_i'] < 0.1) & (d['PSFMAGERR_z'] < 0.1)
	d = d[idx]
	print '# %d stars have acceptable SDSS photometry.' % len(d)
	
	color = d['PSFMAG_g'] - d['PSFMAG_r']
	xlabel = r'$g - r$'
	
	fig = plt.figure(figsize=(8,8))
	
	ax = fig.add_subplot(2,2,1)
	img = density_scatter(ax, color, d['logg'], nbins=(100,100))
	ax.set_title(r'$\mathrm{Complete \ Catalog}$', fontsize=20)
	ax.set_ylabel(r'$\mathrm{log} \, g$', fontsize=20)
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	
	ax = fig.add_subplot(2,2,2)
	idx = (d['feh'] > -0.5) & (d['feh'] < 0.) & (d['teff'] > 5000.) & (d['teff'] < 5500.)
	img = density_scatter(ax, color[idx], d['logg'][idx], nbins=(20,20), threshold=3)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax = fig.add_subplot(2,2,3)
	idx = (d['feh'] > -0.5) & (d['feh'] < 0.) & (d['teff'] > 6000.) & (d['teff'] < 6500.)
	img = density_scatter(ax, color[idx], d['logg'][idx], nbins=(20,20), threshold=3)
	ax.set_xlabel(xlabel, fontsize=20)
	ax.set_ylabel(r'$\mathrm{log} \, g$', fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax = fig.add_subplot(2,2,4)
	idx = (d['feh'] > -0.5) & (d['feh'] < 0.) & (d['teff'] > 7000.) & (d['teff'] < 7500.)
	img = density_scatter(ax, color[idx], d['logg'][idx], nbins=(20,20), threshold=3)
	ax.set_xlabel(xlabel, fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	fig.subplots_adjust(wspace=0, hspace=0)

def T_vs_logg(d):
	fig = plt.figure(figsize=(8,8))
	
	ax = fig.add_subplot(2,2,1)
	img = density_scatter(ax, np.log10(d['teff']), d['logg'], nbins=(100,100))
	ax.set_title(r'$\mathrm{Complete \ Catalog}$', fontsize=20)
	ax.set_ylabel(r'$\mathrm{log} \, g$', fontsize=20)
	xlim = list(ax.get_xlim()); tmp = xlim[0]; xlim[0] = xlim[1]; xlim[1] = tmp
	ylim = ax.get_ylim()
	ax.set_xlim(xlim)
	
	ax = fig.add_subplot(2,2,2)
	idx = d['feh'] < -2.
	img = density_scatter(ax, np.log10(d['teff'])[idx], d['logg'][idx], nbins=(80,80))
	ax.set_title(r'$\left[ \mathrm{Fe} / \mathrm{H} \right] < -2$', fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax = fig.add_subplot(2,2,3)
	idx = (d['feh'] < -1.) & (d['feh'] > -2.)
	img = density_scatter(ax, np.log10(d['teff'])[idx], d['logg'][idx], nbins=(80,80))
	ax.set_title(r'$-2 < \left[ \mathrm{Fe} / \mathrm{H} \right] < -1$', fontsize=20)
	ax.set_xlabel(r'$\mathrm{log} \, T_{\mathrm{eff}}$', fontsize=20)
	ax.set_ylabel(r'$\mathrm{log} \, g$', fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax = fig.add_subplot(2,2,4)
	idx = d['feh'] > -1.
	img = density_scatter(ax, np.log10(d['teff'])[idx], d['logg'][idx], nbins=(80,80))
	ax.set_title(r'$\left[ \mathrm{Fe} / \mathrm{H} \right] > -1$', fontsize=20)
	ax.set_xlabel(r'$\mathrm{log} \, T_{\mathrm{eff}}$', fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

def abs_mag(d):
	print '# %d stars have absolute magnitude estimates.' % np.sum(d['Mi'] > -999.)
	
	dist_V = 1000. * d['DISTV_KPC']
	dist_i = d['DISTi']
	idx = (dist_V > 0.) & (dist_i > 0.)
	diff = (dist_V - dist_i)[idx]
	print diff
	print np.mean(diff)
	print np.max(dist_V[idx]), np.mean(dist_V[idx]), np.median(dist_V[idx])
	
	print np.sqrt(np.sum(diff*diff)) / float(diff.size - 1.)

def color_color(d):
	idx = (d['PSFMAG_u'] > 0.) & (d['PSFMAG_g'] > 0.) & (d['PSFMAG_r'] > 0.) & (d['PSFMAG_i'] > 0.) & (d['PSFMAG_z'] > 0.)
	idx = idx & (d['PSFMAGERR_u'] < 0.1) & (d['PSFMAGERR_g'] < 0.1) & (d['PSFMAGERR_r'] < 0.1) & (d['PSFMAGERR_i'] < 0.1) & (d['PSFMAGERR_z'] < 0.1)
	d = d[idx]
	print '# %d stars have acceptable SDSS photometry.' % len(d)
	
	color1 = d['PSFMAG_u'] - d['PSFMAG_g']
	color2 = d['PSFMAG_g'] - d['PSFMAG_r']
	xlabel = r'$u - g$'
	ylabel = r'$g - r$'
	
	fig = plt.figure(figsize=(8,8))
	
	ax = fig.add_subplot(2,2,1)
	img = density_scatter(ax, color1, color2, nbins=(100,100))
	ax.set_title(r'$\mathrm{Complete \ Catalog}$', fontsize=20)
	ax.set_ylabel(ylabel, fontsize=20)
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	
	ax = fig.add_subplot(2,2,2)
	idx = (d['feh'] > -3.5) & (d['feh'] < -2.0)# & (d['teff'] > 5000.) & (d['teff'] < 5500.)
	print '# %d stars in 1st volume element.' % np.sum(idx)
	img = density_scatter(ax, color1[idx], color2[idx], nbins=(80,80), threshold=3)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax = fig.add_subplot(2,2,3)
	idx = (d['feh'] > -2.0) & (d['feh'] < -1.0)# & (d['teff'] > 6000.) & (d['teff'] < 6500.)
	print '# %d stars in 2nd volume element.' % np.sum(idx)
	img = density_scatter(ax, color1[idx], color2[idx], nbins=(80,80), threshold=3)
	ax.set_xlabel(xlabel, fontsize=20)
	ax.set_ylabel(ylabel, fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	ax = fig.add_subplot(2,2,4)
	idx = (d['feh'] > -1.0) & (d['feh'] < 0.0)# & (d['teff'] > 7000.) & (d['teff'] < 7500.)
	print '# %d stars in 3rd volume element.' % np.sum(idx)
	img = density_scatter(ax, color1[idx], color2[idx], nbins=(80,80), threshold=3)
	ax.set_xlabel(xlabel, fontsize=20)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	
	fig.subplots_adjust(wspace=0, hspace=0)

def grid_models(d):
	# Select stars with reasonable SDSS photometry
	idx = (d['PSFMAG_u'] > 0.) & (d['PSFMAG_g'] > 0.) & (d['PSFMAG_r'] > 0.) & (d['PSFMAG_i'] > 0.) & (d['PSFMAG_z'] > 0.)
	idx = idx & (d['PSFMAGERR_u'] < 0.1) & (d['PSFMAGERR_g'] < 0.1) & (d['PSFMAGERR_r'] < 0.1) & (d['PSFMAGERR_i'] < 0.1) & (d['PSFMAGERR_z'] < 0.1)
	d = d[idx]
	print '# %d stars have acceptable SDSS photometry.' % len(d)
	
	# Determine stellar colors
	color = np.empty((4,len(d)), dtype=np.float32)
	color[0] = d['PSFMAG_u'] - d['PSFMAG_g']
	color[1] = d['PSFMAG_g'] - d['PSFMAG_r']
	color[2] = d['PSFMAG_r'] - d['PSFMAG_i']
	color[3] = d['PSFMAG_i'] - d['PSFMAG_z']
	
	# Grid spectra by stellar parameters
	N_Teff = 30
	N_FeH = 10
	N_logg = 2
	Teff = np.linspace(np.min(d['teff'])-1.e-10, np.max(d['teff'])+1.e-10, N_Teff)
	FeH = np.linspace(np.min(d['feh'])-1.e-10, np.max(d['feh'])+1.e-10, N_FeH)
	logg = np.linspace(np.min(d['logg'])-1.e-10, np.max(d['logg'])+1.e-10, N_logg)
	
	index_grid = np.mgrid[0:N_Teff, 0:N_FeH, 0:N_logg]
	Teff_grid = Teff[index_grid[0]].flatten()
	FeH_grid = FeH[index_grid[1]].flatten()
	logg_grid = logg[index_grid[2]].flatten()
	param_grid = np.vstack((Teff_grid, FeH_grid, logg_grid)).T
	#for i in range(100):
	#	print param_grid[i]
	
	idx_Teff = np.digitize(d['teff'], Teff) - 1
	idx_FeH = np.digitize(d['feh'], FeH) - 1
	idx_logg = np.digitize(d['logg'], logg) - 1
	grid_index = N_logg*N_FeH * idx_Teff + N_logg * idx_FeH + idx_logg
	mean_color = np.zeros((4,N_Teff*N_FeH*N_logg), dtype=np.float32)
	#mean_color.fill(np.nan)
	#weight = np.empty(50*20*20, dtype=np.float32)
	#weight.fill(0.)
	
	print np.min(idx_Teff), np.max(idx_Teff)
	print np.min(idx_FeH), np.max(idx_FeH)
	print np.min(idx_logg), np.max(idx_logg)
	
	for n,i in iterators.index_by_key(grid_index):
		for j in range(4):
			mean_color[j,n] = np.mean(color[j,i])
		#weight[n] = 1.
	
	#mean_color.shape = (4, 50, 20, 20)
	#weight.shape = (50, 20, 20)
	
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(1,1,1)
	ax.plot(mean_color[0], mean_color[1], '.')
	
	# Interpolate colors over grid
	color_interp = []
	for j in range(4):
		idx = ~np.isnan(mean_color[j])
		fuzz = 0.001 * (np.random.random((np.sum(idx), 3)) - 0.5)
		print param_grid[idx]
		color_interp.append(scipy.interpolate.LinearNDInterpolator(param_grid, mean_color[j]))
	
	# Plot colors
	fig = plt.figure(figsize=(8,8))
	
	for i,FeH_i in enumerate([-2.5, -1.5, -0.5, 0.]):
		ax = fig.add_subplot(2,2,i+1)
		x = np.empty((200,3), dtype=np.float32)
		x[:,0] = np.linspace(np.min(d['teff']), np.max(d['teff']), 200)
		x[:,1] = FeH_i
		x[:,2] = np.mean(d['logg'])
		#print x
		ug = color_interp[0](x)
		gr = color_interp[1](x)
		idx = (ug != 0.) & (gr != 0.)
		#print ug
		#print gr
		ax.plot(ug[idx], gr[idx])
	
	return color_interp

def spline_colors(d):
	# Select stars with reasonable SDSS photometry
	idx = (d['PSFMAG_u'] > 0.) & (d['PSFMAG_g'] > 0.) & (d['PSFMAG_r'] > 0.) & (d['PSFMAG_i'] > 0.) & (d['PSFMAG_z'] > 0.)
	idx = idx & (d['PSFMAGERR_u'] < 0.1) & (d['PSFMAGERR_g'] < 0.1) & (d['PSFMAGERR_r'] < 0.1) & (d['PSFMAGERR_i'] < 0.1) & (d['PSFMAGERR_z'] < 0.1)
	d = d[idx]
	print '# %d stars have acceptable SDSS photometry.' % len(d)
	
	# Sort data by temperature
	idx = np.argsort(d['teff'])
	d = d[idx]
	
	# Determine stellar colors
	color = np.empty((4,len(d)), dtype=np.float32)
	color[0] = d['PSFMAG_u'] - d['PSFMAG_g']
	color[1] = d['PSFMAG_g'] - d['PSFMAG_r']
	color[2] = d['PSFMAG_r'] - d['PSFMAG_i']
	color[3] = d['PSFMAG_i'] - d['PSFMAG_z']
	
	# Assign weight to each data point in color space
	weight = np.empty((4, len(d)), dtype=np.float32)
	weight[0] = 1. / (d['PSFMAGERR_u']*d['PSFMAGERR_u'] + d['PSFMAGERR_g']*d['PSFMAGERR_u'])
	weight[1] = 1. / (d['PSFMAGERR_g']*d['PSFMAGERR_g'] + d['PSFMAGERR_r']*d['PSFMAGERR_r'])
	weight[2] = 1. / (d['PSFMAGERR_r']*d['PSFMAGERR_r'] + d['PSFMAGERR_i']*d['PSFMAGERR_i'])
	weight[3] = 1. / (d['PSFMAGERR_i']*d['PSFMAGERR_i'] + d['PSFMAGERR_z']*d['PSFMAGERR_z'])
	
	# Find duplicate parameter entries
	mask_base = (np.diff(d['teff']) == 0.) & (np.diff(d['feh']) == 0.) & (np.diff(d['logg']) == 0.)
	print np.sum(mask_base)
	
	# Construct spline for each color
	color_spl = []
	for j in range(4):
		mask = np.logical_not((np.diff(color[j]) == 0.) & mask_base)
		print j, np.sum(mask)
		tmp_spl,u = scipy.interpolate.splprep([color[j][mask], d['teff'][mask], d['feh'][mask], d['logg'][mask]], w=weight[j][mask], s=70000., k=3, nest=-1)
		color_spl.append(tmp_spl)
	
	f = open('testspline', 'wb')
	pickle.dump(color_spl, f)
	f.close()
	
	return color_spl


def main():
	segue_fname = 'segue_spec.fit'
	d,h = pyfits.getdata(segue_fname, header=True)
	print '# %d stars with SEGUE spectra.' % len(d)
	
	spline_colors(d)
	
	#mplib.rc('text', usetex=True)
	#color_color(d)
	#grid_models(d)
	#gr_vs_logg(d)
	#T_vs_logg(d)
	#plt.show()
	
	#abs_mag(d)
	
	return 0

if __name__ == '__main__':
	main()

