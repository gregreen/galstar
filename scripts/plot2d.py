#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       plot2d.py
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
import matplotlib as mplib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse, sys


# Load most likely points from files formatted as:
# 	x	y	p(x,y)
def load_MLs(fn_list):
	x_ML = np.empty(len(fn_list))
	y_ML = np.empty(len(fn_list))
	for i,fn in enumerate(fn_list):
		x,y,p = np.loadtxt(fn, usecols=(0, 1, 2), unpack=True)
		if x_list
		n = np.argmax(p)
		x_ML[i] = x[n]
		y_ML[i] = y[n]
	return x_ML, y_ML

def plot_MLs(x_ML, y_ML, xname, yname):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x_ML, y_ML)
	ax.set_xlabel(r'$%s$'%xname, fontsize=14)
	ax.set_ylabel(r'$%s$'%yname, fontsize=14)
	ax.set_title(r'$\mathrm{Maximum \ Likelihoods}$', fontsize=18)
	return fig


def main():
	parser = argparse.ArgumentParser(prog='plot2d', description='Make a scatter plot of the Maximum Likelihoods', add_help=True)
	parser.add_argument('--files', nargs='+', type=str, required=True, help='Input posterior distributions')
	parser.add_argument('--axname', nargs=2, type=str, required=True, help='Name of x- & y-axes, respectively')
	parser.add_argument('--output', type=str, required=True, help='Output image filename (with extension)')
	
	# Handle presence or absence of 'python' in commandline arguments
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Make sure the output filename has an extension
	output_fn = values.output
	if '.' not in output_fn:
		output_fn += '.png'
	
	# General plotting options
	mplib.rc('text', usetex='True')
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	x_ML, y_ML = load_MLs(values.files)
	fig = plot_MLs(x_ML, y_ML, values.axname[0], values.axname[1])
	fig.savefig(output_fn, dpi=300, transparent=False)
	
	return 0

if __name__ == '__main__':
	main()

