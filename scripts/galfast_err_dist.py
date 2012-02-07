#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       galfast_err_dist.py
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
import numpy as np
from os.path import abspath
from galfast_utils import SED, get_objects
from get_colors import get_abs_mags, calc_abs_mag

import matplotlib.pyplot as plt
import matplotlib as mplib

#from random import random


def main():
	parser = argparse.ArgumentParser(prog='galfast_err_dist.py', description='Test distribution of errors in absolute magnitudes from galfast.', add_help=True)
	parser.add_argument('catalog', type=str, help='Galfast catalog (as a FITS file)')
	parser.add_argument('--maxmag', type=float, nargs='+', help='Faintest magnitude to include')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Set up the magnitude cutoff
	mag_cutoff = None
	if values.maxmag != None:
		if len(values.maxmag) == 1:
			mag_cutoff = np.empty(5, dtype=float)
			mag_cutoff.fill(values.maxmag)
		elif len(values.maxmag) == 5:
			mag_cutoff = np.array(values.maxmag)
		else:
			print 'Invalid # of arguments for maxmag. Enter either a unform magnitude limit, or a separate limit for each band (5).'
			return 1
	
	# Load apparent magnitudes (w/ and w/o photometric errors) and theoretical errors
	fname = abspath(values.catalog)
	ra_dec, obs_mags, LSST_mags, errs, params = get_objects(fname)
	N = len(obs_mags)
	
	# Load the SED catalog
	SED_catalog = SED()
	#for j in range(N):
	#	print calc_abs_mag(LSST_mags[j], params[j]) - SED_catalog(params[j][2], params[j][3])[:-1]
	
	# Determined normalized errors
	score = [np.empty(N, dtype=float) for i in range(5)]		# Normalized difference between observed and true apparent magnitudes
	mag_diff = [np.empty(N, dtype=float) for i in range(5)]		# Difference between galfast and catalog magnitudes
	for i in range(N):
		tmp_diff = calc_abs_mag(LSST_mags[i], params[i]) - SED_catalog(params[i][2], params[i][3])[:-1]
		for j in range(5):
			score[j][i] = (obs_mags[i,j] - LSST_mags[i,j]) / errs[i,j]
			mag_diff[j][i] = tmp_diff[j]
	
	# Filter out faint stars
	if values.maxmag != None:
		for i in range(5):
			filtmag = np.empty(N, dtype=bool)
			filtmag.fill(True)
			for k in range(N):
				filtmag[k] = (LSST_mags[k,:].max() <= mag_cutoff[i])
			score[i] = score[i][filtmag]
			mag_diff[i] = mag_diff[i][filtmag]
	
	# Take log of differences
	log_mag_diff = []
	for i in range(5):
		log_mag_diff.append(np.log(mag_diff[i]))
		idx = np.isfinite(log_mag_diff[i])
		#idx = np.isneginf(log_mag_diff[i])
		#idx = np.logical_or(idx, np.isnan(log_mag_diff[i]))
		#idx = np.logical_not(idx)
		min_log_diff = min(log_mag_diff[i][idx])
		print min_log_diff
		for j in range(len(idx)):
			if not idx[j]:
				log_mag_diff[i][j] = min_log_diff
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=False)
	
	# Create histogram of scores
	fig_score = plt.figure(figsize=(8.5,11))
	fig_score.suptitle(r'$\mathrm{Distribution \ of \ Scores}$', fontsize=22, y=0.96)
	fig_diff = plt.figure(figsize=(8.5,11))
	fig_diff.suptitle(r'$\mathrm{Distribution \ of \ Log \ Differences}$', fontsize=22, y=0.96)
	name = ['u', 'g', 'r', 'i', 'z']
	color = ['cyan', 'g', 'r', 'magenta', 'purple']
	for i in range(5):
		ax = fig_score.add_subplot(3,2,i+1)
		ax.set_title(r'$%s$' % name[i], fontsize=20)
		ax.set_xlabel(r'$z$', fontsize=18)
		ax.set_ylabel(r'$N$', fontsize=18)
		ax.hist(score[i], bins=30, color=color[i])
		#
		ax = fig_diff.add_subplot(3,2,i+1)
		ax.set_title(r'$%s$' % name[i], fontsize=20)
		ax.set_xlabel(r'$\Delta m$', fontsize=18)
		ax.set_ylabel(r'$N$', fontsize=18)
		ax.hist(log_mag_diff[i], bins=30, color=color[i])
	fig_score.subplots_adjust(hspace=0.3, wspace=0.25)
	fig_diff.subplots_adjust(hspace=0.3, wspace=0.25)
	
	outfile = fname.replace('.fits', '')
	fig_score.savefig(outfile + '_score.png', dpi=150)
	fig_diff.savefig(outfile + '_diff.png', dpi=150)
	plt.show()
	
	
	# Intitialize the SED catalog
	#ugrizy = SED()
	# Load the absolute magnitudes from galfast
	#abs_mags = get_abs_mags(values.catalog)
	'''for i in range(10):
		Mr = -1. + 29.*random()
		FeH = -2.5 + 2.5*random()
		tmp = ugrizy(Mr, FeH)
		print '(%.3f, %.2f): %.3f %.3f %.3f %.3f %.3f %.3f' % (Mr, FeH, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])'''
	
	return 0

if __name__ == '__main__':
	main()

