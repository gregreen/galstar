#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       plot_hist.py
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

# TODO: Generalize background subtraction to all dimensions

import numpy as np
import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from math import cos, sin, log, exp, pi, sqrt
from scipy.integrate import quad
import argparse, sys

from astroutils import parse_dhms, parse_RA, parse_DEC


def plot_hist(ax, bin_x, p_x, var_name):
	ax.fill_between(bin_x, 0, p_x, alpha=0.8)
	ax.set_xlabel(r'$%s$'%var_name, fontsize=14)
	ax.set_ylabel(r'$\mathrm{d}n / \mathrm{d}%s$'%var_name, fontsize=14)
	ax.set_title(r'$\mathrm{Stacked} \ p(%s)$'%var_name, fontsize=18)
	return ax.get_xlim(), ax.get_ylim()

def plot_MLs(ax, fn_list, var_name, col=0):
	x_ML = load_MLs(fn_list, col)
	ax.hist(x_ML, 50, alpha=0.8)
	ax.set_xlabel(r'$%s$'%var_name, fontsize=14)
	ax.set_ylabel(r'$N$', fontsize=14)
	ax.set_title(r'$\mathrm{Binned \ Maximum \ Likelihoods}$', fontsize=18)
	return ax.get_xlim(), ax.get_ylim()

def sum_hist(fn_list, col=0, subtract=(None,None,None)):
	bin_x, p_x = load_hist(fn_list[0], col)
	for i,fn in enumerate(fn_list[1:]):
		bin_x_i, p_x_i = load_hist(fn, col)
		if np.any(bin_x_i != bin_x):
			print "Bin positions for %s do not match!" % fn
		if(p_x_i[0] == p_x_i[0]): p_x += p_x_i
	if subtract != (None, None,None):
		l = pi/180.*subtract[0]
		b = pi/180.*subtract[1]
		r = pi/180.*subtract[2]
		cos_l, sin_l, cos_b, sin_b = cos(l), sin(l), cos(b), sin(b)
		f = lambda x: dn_dDM(cos_l, sin_l, cos_b, sin_b, x, radius=r)
		norm = quad(f, 0.01, 25., epsrel=1.e-5)[0]
		print norm
		for i in range(len(p_x)):
			p_x[i] -= dn_dDM(cos_l, sin_l, cos_b, sin_b, bin_x[i], radius=r)# * float(N)/norm
	return bin_x, p_x

def load_hist(fn, col=0):
	col_0, col_1, p = np.loadtxt(fn, usecols=(0, 1, 2), unpack=True)
	data = [col_0, col_1]
	# Filter zero values from p
	p_filtered = p.copy()
	p_min = p.min()
	for i in range(len(p_filtered)):
		if p_filtered[i] == p_min: p_filtered[i] = -999.
	# Determine sample positions
	bin_x = np.array(list(set(data[col])))
	bin_x.sort()
	# Marginalize over data[-col]
	p_x = np.zeros(len(bin_x), dtype=float)
	for i,x in enumerate(bin_x):
		idx = (data[col] == x)
		p_x[i] = sum(np.exp(p_filtered[idx]))
	# Normalize probability to unity
	#p_x_sum = sum(p_x)
	#for i in range(len(p_x)):
	#	p_x[i] /= p_x_sum
	return bin_x, p_x

def load_MLs(fn_list, col=0):
	x_ML = np.empty(len(fn_list))
	bin_x, p_x = load_hist(fn_list[0], col)
	x_ML[0] = bin_x[np.argmax(p_x)]
	for i,fn in enumerate(fn_list[1:]):
		bin_x_i, p_x_i = load_hist(fn, col)
		if np.any(bin_x_i != bin_x):
			print "Bin positions for %s do not match!" % fn
		x_ML[i+1] = bin_x[np.argmax(p_x_i)]
	x_ML_sorted = x_ML.copy()
	x_ML_sorted.sort()
	N = len(x_ML_sorted)
	print '25th percentile: %1.2f' % x_ML_sorted[N/4]
	print 'Median: %1.2f' % x_ML_sorted[N/2]
	print '75th percentile: %1.2f' % x_ML_sorted[3*N/4]
	return x_ML

def Cartesian_coords(cos_l, sin_l, cos_b, sin_b, DM):
	R0 = 8000.
	d = 10.**(DM/5. + 1.)
	x = R0 - cos_l*cos_b*d
	y = -sin_l*cos_b*d
	z = sin_b*d
	return x, y, z

def dn_dDM(cos_l, sin_l, cos_b, sin_b, DM, radius=1.):
	rho_0 = 0.0058/1.e9/5.
	R0 = 8000.
	Z0 = 25.
	H1, L1 = 2150., 245.
	f, H2, L2 = 0.13, 3261., 743.
	fh, qh, nh = 0.0051, 0.64, -2.77
	x,y,z = Cartesian_coords(cos_l, sin_l, cos_b, sin_b, DM)
	r = sqrt(x*x + y*y)
	rho_thin = exp(-(abs(z+Z0) - abs(Z0))/H1 - (r-R0)/L1)
	rho_thick = f*exp(-(abs(z+Z0) - abs(Z0))/H2 - (r-R0)/L2)
	rho_halo = fh * ((r*r + (z/qh)*(z/qh))/(R0*R0))**(nh/2.)
	rho = rho_0  * (rho_thin + rho_thick + rho_halo)
	volume = (pi*radius**2.) * (1000.*2.30258509/5.) * exp((3.*2.30258509/5.)*DM)
	return rho * volume

def main():
	parser = argparse.ArgumentParser(prog='plot_hist', description='Plot stacked pdfs marginalized over one dimension and binned Maximum Likelihoods', add_help=True)
	parser.add_argument('files', nargs='+', type=str, help='Input posterior distributions')
	parser.add_argument('--xname', type=str, required=True, help='Name of x-axis')
	parser.add_argument('--output', type=str, required=True, help='Output image filename (with extension)')
	parser.add_argument('--xaxis', type=int, default=0, choices=(0,1), help='Column to use as x-axis')
	parser.add_argument('--subtract-background', nargs=3, type=float, default=(None, None, None), help='Remove background, assuming line-of-sight (l, b, radius), in degrees')
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--MLonly', action='store_true', help='Only plot histogram of binned Maximum Likelihoods')
	group.add_argument('--Pxonly', action='store_true', help='Only plot stacked pdfs')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Make sure the output filename has an extension
	output_fn = values.output
	if '.' not in output_fn:
		output_fn += '.png'
	
	mplib.rc('text', usetex='True')
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	ax = []
	if values.Pxonly:		# Plot continuous marginalized probabilities
		fig = plt.figure()
		ax.append(fig.add_subplot(111))
		bin_x, p_x = sum_hist(values.files, col=values.xaxis, subtract=values.subtract_background)
		xlim, ylim = plot_hist(ax[0], bin_x, p_x, values.xname)
	elif values.MLonly:		# Plot histogram of max. likelihoods
		fig = plt.figure()
		ax.append(fig.add_subplot(111))
		xlim, ylim = plot_MLs(ax[0], values.files, values.xname, col=values.xaxis)
	else:					# Plot both
		fig = plt.figure(figsize=(8.5,11.))
		ax.append(fig.add_subplot(211))
		ax.append(fig.add_subplot(212))
		fig.subplots_adjust(left=0.16, right=0.88, hspace=0.25)
		bin_x, p_x = sum_hist(values.files, col=values.xaxis, subtract=values.subtract_background)
		xlim1, ylim1 = plot_hist(ax[0], bin_x, p_x, values.xname)
		xlim2, ylim2 = plot_MLs(ax[1], values.files, values.xname, col=values.xaxis)
		xlim = (min(xlim1[0],xlim2[0]), max(xlim1[1],xlim2[1]))
		ax[0].set_xlim(xlim)
		ax[1].set_xlim(xlim)
		for ax_i in ax:		# Adjust title position
			x,y = ax_i.title.get_position()
			ax_i.title.set_y(y+0.02)
	
	# Format ticks
	dx_major = int(round((xlim[1]-xlim[0])/4))
	if dx_major >= 8:
		dx_minor = 2
	elif dx_major >= 3:
		dx_minor = 1
	else:
		dx_minor = 0.25
	loc_major = MultipleLocator(dx_major)
	loc_minor = MultipleLocator(dx_minor)
	fmt = FormatStrFormatter(r'$%d$')
	for ax_i in ax:
		ax_i.xaxis.set_major_locator(loc_major)
		ax_i.xaxis.set_minor_locator(loc_minor)
		ax_i.xaxis.set_major_formatter(fmt)
		for tick in ax_i.xaxis.get_major_ticks() + ax_i.xaxis.get_minor_ticks() + ax_i.yaxis.get_major_ticks() + ax_i.yaxis.get_minor_ticks():
			tick.tick2On = False
		ax_i.grid(which='minor', alpha=0.5)
	
	fig.savefig(output_fn, transparent=False, dpi=300)
	
	return 0

if __name__ == '__main__':
	main()

