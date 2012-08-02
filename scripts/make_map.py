#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#       make_map.py
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

import matplotlib as mplib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt

import numpy as np
import healpy as hp

import pyfits
import gzip

import sys, argparse
import os
from os.path import abspath, isdir

import healpix_utils as hputils
import iterators


def load_reddening(fname):
	'''
	Load the pixels contained in the file(s) given by <fname>,
	returning the reddening profile, along with the healpix
	pixel number and chi^2/d.o.f. of the fit for each line of
	sight.
	
	Input:
		fname		(string or list of strings)
	
	Output:
		mu_anchors	(list of flat np.float64 arrays)
		Ar_anchors	(list of flat np.float64 arrays)
		pix_index	(list of integers)
		chi2dof		(listo f floats)
	'''
	
	if type(fname) is str:
		fname = [fname]
	
	mu_anchors_list = []
	Ar_anchors_list = []
	pix_index_list = []
	chi2dof_list = []
	
	# Store (DM, Ar) fit for each healpix pixel
	for filename in fname:
		#print 'Opening %s ...' % filename
		f = open(abspath(filename), 'rb')
		
		while True:
			try:
				pix_index = np.fromfile(f, dtype=np.uint64, count=1)[0]
				N_stars = np.fromfile(f, dtype=np.uint32, count=1)[0]
				measure = np.fromfile(f, dtype=np.float64, count=1)[0]
				success = np.fromfile(f, dtype=np.uint16, count=1)[0]
				N_regions = np.fromfile(f, dtype=np.uint16, count=1)[0]
				line_int = np.fromfile(f, dtype=np.float64, count=N_stars)
				mu_anchors = np.fromfile(f, dtype=np.float64, count=N_regions+1)
				Ar_anchors = np.fromfile(f, dtype=np.float64, count=N_regions+1)
				
				
				#print 'pixel index: %d' % pix_index
				#print '# of stars: %d' % N_stars
				'''
				print 'chisq: %.2f (%.2f / d.o.f.)' % (measure, measure / float(N_stars - int(N_regions) - 1))
				print 'success: %d' % success
				print '# of regions: %d' % N_regions
				'''
				'''
				print 'line integrals:'
				print line_int
				print 'mu anchors:'
				print mu_anchors
				print 'A_r anchors:'
				print Ar_anchors
				'''
				#print ''
				
				mu_anchors_list.append(mu_anchors)
				Ar_anchors_list.append(Ar_anchors)
				pix_index_list.append(pix_index)
				chi2dof_list.append(measure / float(N_stars - int(N_regions) - 1))
				
			except:
				f.close()
				break
		
		f.close()
	
	pix_index_list = np.array(pix_index_list, dtype=np.uint64)
	
	return mu_anchors_list, Ar_anchors_list, np.array(pix_index_list, dtype=np.uint64), np.array(chi2dof_list, dtype=np.float64)


def eval_Ar_old(mu_anchors, Ar_anchors, pix_index, mu, nside=512):
	if type(mu_anchors) is not list:
		mu_anchors = [mu_anchors]
	if type(Ar_anchors) is not list:
		Ar_anchors = [Ar_anchors]
	if len(mu_anchors) != len(Ar_anchors):
		raise Exception('mu_anchors and Ar_anchors must contain same # of elements.')
	if len(mu_anchors) != len(pix_index):
		raise Exception('mu_anchors and pix_index must contain same # of elements.')
	
	# Create an empty map for each value of mu
	Ar_map = np.empty((len(mu), 12*nside*nside), dtype=np.float64)
	Ar_map.fill(np.inf)
	
	# Sort the elements in mu
	if len(mu) == 0:
		mu = [mu]
	mu.sort()
	
	# Evaluate each pixel in the maps
	for i, (mu_arr, Ar_arr) in enumerate(zip(mu_anchors, Ar_anchors)):
		n = 0
		for j in xrange(1, len(mu_arr)):
			if mu_arr[j] >= mu[n]:
				slope = (Ar_arr[j] - Ar_arr[j-1]) / (mu_arr[j] - mu_arr[j-1])
				Ar_map[n, pix_index[i]] = Ar_arr[j-1] + slope * (mu[n] - mu_arr[j-1])
				#print '(%d --> %.3f)' % (pix_index[i], Ar_map[n, pix_index[i]])
				#print '\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (mu[n], mu_arr[j-1], mu_arr[j], Ar_arr[j-1], Ar_arr[j])
				
				n += 1
				if n >= len(mu):
					break
	
	return Ar_map


def eval_Ar(mu_anchors, Ar_anchors, pix_index, mu_eval, nside=512):
	if type(mu_anchors) is not list:
		mu_anchors = [mu_anchors]
	if type(Ar_anchors) is not list:
		Ar_anchors = [Ar_anchors]
	if len(mu_anchors) != len(Ar_anchors):
		raise Exception('mu_anchors and Ar_anchors must contain same # of elements.')
	if len(mu_anchors) != len(pix_index):
		raise Exception('mu_anchors and pix_index must contain same # of elements.')
	
	# Create an empty map for each value of mu
	Ar_map = np.empty((len(mu_eval), 12*nside*nside), dtype=np.float64)
	Ar_map.fill(np.inf)
	
	# Sort the elements in mu
	if len(mu_eval) == 0:
		mu_eval = [mu_eval]
	mu_eval.sort()
	
	# Evaluate each pixel in the maps
	for mu_arr, indices in iterators.index_by_unsortable_key(mu_anchors):
		Ar_arr = np.empty((len(mu_arr), len(indices)), dtype=np.float64)
		for i in indices:
			Ar_arr[:,i] = Ar_anchors[i]
		
		n = 0
		j = 1
		pix = pix_index[indices]
		while j <= len(mu_arr):
			if mu_arr[j] >= mu_eval[n]:
				slope = (Ar_arr[j] - Ar_arr[j-1]) / (mu_arr[j] - mu_arr[j-1])
				Ar_map[n, pix] = Ar_arr[j-1] + slope * (mu_eval[n] - mu_arr[j-1])
				
				#print '%.3f <= %.3f <= %.3f' % (mu_arr[j-1], mu_eval[n], mu_arr[j])
				#print 'Delta_mu = %.3g' % (mu_eval[n] - mu_arr[j-1])
				#print 'slope:'
				#print slope
				#print ''
				
				n += 1
				if n >= len(mu_eval):
					break
			else:
				j += 1
	
	return Ar_map



def write_maps(fname, maps, mu, nest=True):
	'''
	Write extinction map to a FITS file.
	
	Input:
	    fname  output filename
	    maps   set of extinction maps at different distances
	    mu     distance modulus of each extinction map
	    nest   True if maps are given with nested ordering. False if ring
	           ordering is used.
	'''
	
	if len(maps.shape) == 1:
		maps.shape = (1, maps.shape[0])
	elif len(maps.shape) != 2:
		raise Exception('<maps> has invalid shape.')
	if len(mu) != len(maps):
		raise Exception('<mu> must have same length as <maps>.')
	
	'''
	cols = []
	cols.append(pyfits.Column(name='MU', format='%dD' % len(mu), array=mu))
	for i, m in enumerate(maps):
		cols.append(pyfits.Column(name='A_R %d' % i, format='D', array=m))
	
	tbhdu = pyfits.new_table(cols)
	tbhdu.header.update('NESTED', nest, 'Healpix ordering scheme.')
	tbhdu.header.update('NSIDE', hp.npix2nside(maps.shape[1]), 'Healpix nside parameter.')
	
	tbhdu.writeto(fname, clobber=True)
	'''
	
	hdu = []
	hdu.append(pyfits.PrimaryHDU(mu))
	for m in maps:
		hdu.append(pyfits.ImageHDU(m))
	hdulist = pyfits.HDUList(hdu)
	hdulist.writeto(fname, clobber=True)


def read_maps(fname):
	'''
	Read in extinction map from a FITS file.
	
	Input:
	    fname  input filename
	'''
	
	d,h = pyfits.read(fname)
	# TODO: Finish this function




def main():
	parser = argparse.ArgumentParser(prog='make_map.py', description='Generate a map from the given reddening files generated by fit_pdfs.py.', add_help=True)
	parser.add_argument('input', type=str, nargs='+', help='Reddening files generated by fit_pdfs.py, or directory containing input as .dat files.')
	parser.add_argument('-n', '--nside', type=int, default=512, help='Healpix nside parameter.')
	parser.add_argument('-lb', '--lb_bounds', type=float, nargs=4, default=(0., 360., -90., 90.), help='(l_min, l_max, b_min, b_max).')
	parser.add_argument('-mol', '--mollweide', action='store_true', help='Use Mollweide projection (incompatible with setting bounds on l and b).')
	parser.add_argument('-sz', '--size', type=int, nargs=2, default=(500,200), help='Dimensions of each image: (xsize, ysize).')
	parser.add_argument('-dpi', '--dpi', type=float, default=150, help='Dots per inch of output image.')
	parser.add_argument('-max', '--max_Ar', type=float, default=None, help='Maximum Ar in color scale.')
	parser.add_argument('-nst', '--nest', action='store_true', help='Maps are stored in nested ordering scheme.')
	parser.add_argument('-d', '--diff', action='store_true', help='Show differential extinction at each distance modulus.')
	parser.add_argument('-mu', '--mu', type=float, nargs='+', default=(5., 6.5, 8., 9.5, 11., 12.5, 14., 15.5, 17.), help='Distance moduli at which to show reddening.')
	parser.add_argument('-rc', '--rowcol', type=int, nargs=2, default=(3,3), help='Distance moduli at which to show reddening.')
	parser.add_argument('-fig', '--figsize', type=float, nargs=2, default=(9,5), help='Width and height of figure, in inches.')
	parser.add_argument('-po', '--plotout', type=str, default=None, help='Output filename for plot.')
	parser.add_argument('-fo', '--fitsout', type=str, default=None, help='Output filename (of type FITS) for reddening map.')
	parser.add_argument('-sh', '--show', action='store_true', help='Show plot.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	#if values.rowcol[0] * values.rowcol[1] != len(values.mu):
	#	print 'Number of rows and columns must correspond to number of distance moduli in plot.'
	#	return 1
	
	# Load in pixels
	input_files = None
	if (len(values.input) == 1) and isdir(values.input[0]):
		input_files = []
		for f in os.listdir(values.input[0]):
			if f.endswith('.dat'):
				input_files.append(os.path.join(values.input[0], f))
		#input_files = filter(lambda f: f.endswith('.dat'), os.listdir(values.input[0]))
	else:
		input_files = values.input
	m = hputils.ExtinctionMap(input_files, FITS=False, nside=values.nside, nested=values.nest)	
	#mu_anchors, Ar_anchors, pix_index, chi2dof = load_reddening(input_files)
	print '%d pixel(s) loaded.' % m.npix()
	
	# Output reddening maps to a FITS file
	if values.fitsout != None:
		print 'Saving extinction map to %s...' % values.fitsout
		m.save(values.fitsout)
	
	# Generate reddening map at a range of distances
	mu_eval = np.array(values.mu, dtype=np.float64)
	Ar_map = m.evaluate(mu_eval)
	
	# Determine maximum A_r within bounds
	pix_index = m.valid_pixels
	theta, phi = hp.pix2ang(values.nside, pix_index, nest=values.nest)
	l, b = hputils.thetaphi2lb(theta, phi)
	pix_mask = (l >= values.lb_bounds[0]) & (l <= values.lb_bounds[1]) & (b >= values.lb_bounds[2]) & (b <= values.lb_bounds[3])
	Ar_map_clipped = Ar_map[-1, pix_index[pix_mask]]
	Ar_map_clipped = Ar_map_clipped[np.isfinite(Ar_map_clipped)]
	Ar_max = np.max(Ar_map_clipped)
	Ar_map_clipped.sort()
	Ar_95pct = Ar_map_clipped[int(0.95*(Ar_map_clipped.size-1.))]
	Ar_99pct = Ar_map_clipped[int(0.99*(Ar_map_clipped.size-1.))]
	del Ar_map_clipped
	print 'bounds: (%.1f %.1f %.1f %.1f)' % (np.min(l), np.max(l), np.min(b), np.max(b))
	#Ar_max = np.max(Ar_map[np.isfinite(Ar_map)])
	print 'max. A_r: %.2f' % Ar_max
	print '95th %%ile: %.2f' % Ar_95pct
	print '99th %%ile: %.2f' % Ar_99pct
	Ar_cutoff = None
	if values.max_Ar != None:
		Ar_cutoff = values.max_Ar
	else:
		Ar_cutoff = Ar_99pct
	
	if values.diff:
		Ar_map[1:,:] = Ar_map[1:,:] - Ar_map[:-1,:]
		Ar_max = np.max(Ar_map[np.isfinite(Ar_map)])
	
	# Handle case where user wants Mollweide projection
	lb_bounds = list(values.lb_bounds)
	if values.mollweide and (values.lb_bounds != [0., 360., -90., 90.]):
		lb_bounds[0] = 0.
		lb_bounds[1] = 360.
		lb_bounds[2] = -90.
		lb_bounds[3] = 90.
		print 'Ignoring option --lb_bounds, as it is incompatible with --mollweide.'
	
	np.seterr(all='ignore')
	mplib.rc('text', usetex=True)
	mplib.rc('axes', grid=False)
	
	center_gal = (lb_bounds[0] == 0.) and (lb_bounds[1] == 360.)
	nrows, ncol = values.rowcol
	
	fig = None
	image = None
	
	for i, m in enumerate(Ar_map):
		print 'Plotting map at mu = %.2f ...' % mu_eval[i]
		
		# Determine whether to make a new figure
		if i % (nrows*ncol) == 0:
			fig = plt.figure(1, figsize=values.figsize, dpi=values.dpi)
			#ax_index = 0
		
		# Plot reddening at this distance modulus
		ax = None
		if values.mollweide:
			ax = fig.add_subplot(nrows, ncol, (i+1) % (nrows*ncol), projection='mollweide')
		else:
			ax = fig.add_subplot(nrows, ncol, (i+1) % (nrows*ncol))
			y, x = np.unravel_index(i % (nrows*ncol), values.rowcol)
			if y != nrows - 1:
				ax.set_xticklabels([])
			if x != 0:
				ax.set_yticklabels([])
		image = hputils.healmap_to_axes(ax, m, values.nside, size=values.size, center_gal=center_gal, lb_bounds=lb_bounds, nest=values.nest, vmin=0., vmax=Ar_cutoff)
		
		# Label images
		x_min, x_max = ax.get_xlim()
		y_min, y_max = ax.get_ylim()
		x, y = x_min + 0.95*(x_max - x_min), y_min + 0.95*(y_max - y_min)
		txt = ax.text(x, y, r'$\mu = %.2f$' % mu_eval[i], color='white', fontsize=14, horizontalalignment='right', verticalalignment='top')
		txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
		
		# Finish figure if filled up
		if ((i+1) % (nrows*ncol) == 0) or (i == len(Ar_map) - 1):
			# Add title
			if values.diff:
				fig.suptitle(r'$\Delta A_r$', fontsize=20, y=0.95)
			else:
				fig.suptitle(r'$A_r$', fontsize=20, y=0.95)
			
			# Add colorbar
			fig.subplots_adjust(wspace=0., hspace=0., left=0.08, right=0.89, top=0.84)
			cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
			cb = fig.colorbar(image, cax=cax)
			
			# Save plot
			if values.plotout != None:
				fname = abspath(values.plotout)
				fmt = '.png'
				if values.plotout.endswith('.png'):
					fmt = '.png'
					fname = fname[:-4]
				elif values.plotout.endswith('.pdf'):
					fmt = '.pdf'
					fname = fname[:-4]
				elif values.plotout.endswith('.eps'):
					fmt = '.eps'
					fname = fname[:-4]
				if len(Ar_map) != 1:
					fname += '.%d' % int(i / (nrows*ncol))
				fname += fmt
				print 'Saving plot to %s' % fname
				fig.savefig(fname, dpi=values.dpi)
				fig.clf()
				plt.close(fig)
	
	if values.show:
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

