#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       comp_stats.py
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


from galfast_utils import *

from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib as mplib
import matplotlib.pyplot as plt

from os.path import abspath
from operator import itemgetter
import sys, argparse


def get_param_index(param_name):
	try:
		return int(param_name)
	except ValueError:
		plower = param_name.lower()
		if plower == 'dm':
			return 0
		elif plower == 'ar':
			return 1
		elif plower == 'mr':
			return 2
		elif plower == 'feh':
			return 3
		else: return None


def main():
	parser = argparse.ArgumentParser(prog='comp_stats.py', description='Compare statistics from two galfast runs.', add_help=True)
	parser.add_argument('--f1', type=str, nargs='+', required=True, help='First set of galstar statistics files')
	parser.add_argument('--f2', type=str, nargs='+', required=True, help='Second set of galstar statistics files')
	parser.add_argument('--converged', action='store_true', help='Filter out nonconverged stars')
	parser.add_argument('--normalize', action='store_true', help='Divide each error by standard deviation')
	parser.add_argument('--output', type=str, help='Output plot filename')
	parser.add_argument('--axes', type=str, nargs=2, default=('DM', 'Ar'), help='Parameters to use as x- and y-axes (default: DM Ar)')
	parser.add_argument('--title', type=str, help='Plot title (in LaTeX markup)')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Determine which parameters to use as x- and y-axes
	axis_index = [None, None]
	for i in range(2):
		axis_index[i] = get_param_index(values.axes[i])
	if None in axis_index:
		print 'Invalid parameter name entered in --axes option.'
		return 1
	
	# Sort both filename lists
	f = [[],[]]
	f[0]= values.f1
	f[1] = values.f2
	for i in range(2):
		z = [(ff, int((ff.split('_')[-1]).split('.')[0])) for ff in f[i]]
		z.sort(key=itemgetter(1))
		for n in range(len(z)):
			f[i][n] = z[n][0]
	
	N = len(f[0])
	
	# Make sure both filename lists are of the same size
	if len(f[1]) < N:
		print '# of files in two sets do not match. Clipping 1st set to match size of 2nd.'
		N = len(f[1])
		f[0] = f[0][0:N]
	elif len(f[1]) > N:
		print '# of files in two sets do not match. Clipping 2nd set to match size of 1st.'
		f[1] = f[1][0:N]
	
	for i in range(N):
		print f[0][i], f[1][i]
	
	# Load stats
	means = np.empty((2, N, 4), dtype=float)
	cov = np.empty((2, N, 4, 4), dtype=float)
	converged = np.empty((2, N), dtype=bool)
	for i in range(N):
		converged[0][i], means[0][i], cov[0][i], tmp1, tmp2 = read_stats(abspath(f[0][i]))
		converged[1][i], means[1][i], cov[1][i], tmp3, tmp4 = read_stats(abspath(f[1][i]))
	
	# Initialize filter
	idx = np.empty(N, dtype=bool)
	idx.fill(True)
	
	# Calculate filter for nonconverged stars
	conv = np.empty((2,N), dtype=bool)
	conv.fill(True)
	if values.converged:
		for i in range(2):
			conv[i] = (converged[i] == True)
	
	# Combine and apply filters
	idx = np.logical_and(conv[0], conv[1])
	for i in range(2):
		means[i] = means[i][idx]
		cov[i] = cov[i][idx]
	print 'Filtered out %d stars.' % (N - len(means[0]))
	N = len(means[0])
	
	# Calculate differences and covariances
	Delta = means[1] - means[0]
	
	for d in Delta:
		print d
	
	print ''
	
	# Calculate errors
	sigma = np.zeros((N, 4), dtype=float)
	for n in range(N):
		for i in range(4):
			for k in range(2):
				sigma[n][i] += cov[k][n][i][i]
			sigma[n][i] = sqrt(sigma[n][i])
		print sigma[n]
	
	# Normalize differences
	if values.normalize:
		for n in range(N):
			for i in range(4):
				Delta[n][i] /= sigma[n][i]
	
	print ''
	
	for d in Delta:
		print d
	
	# Set matplotlib style attributes
	mplib.rc('text',usetex=True)
	mplib.rc('xtick.major', size=6)
	mplib.rc('xtick.minor', size=4)
	mplib.rc('axes', grid=True)
	
	# Set up figure
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	param_name = ['\mu', 'A_r', 'M_r', 'Z']
	if values.normalize:
		ax.set_xlabel(r'$\Delta %s \, / \, \sigma_{%s}$' % (param_name[axis_index[0]], param_name[axis_index[0]]), fontsize=18)
		ax.set_ylabel(r'$\Delta %s \, / \, \sigma_{%s}$' % (param_name[axis_index[1]], param_name[axis_index[1]]), fontsize=18)
	else:
		ax.set_xlabel(r'$\Delta %s$' % param_name[axis_index[0]], fontsize=18)
		ax.set_ylabel(r'$\Delta %s$' % param_name[axis_index[1]], fontsize=18)
	if values.title != None:
		ax.set_title(r'$\mathrm{%s}$' % values.title.replace(' ','\ '), fontsize=22)
	
	# Plot differences
	ax.scatter(Delta[:,axis_index[0]], Delta[:,axis_index[1]])
	
	# Draw axes and set make plot range symmetric
	xlims = max([abs(x) for x in ax.get_xlim()])
	ylims = max([abs(y) for y in ax.get_ylim()])
	ax.plot([-xlims,xlims], [0,0], 'k')
	ax.plot([0,0], [-ylims,ylims], 'k')
	ax.set_xlim(-xlims, xlims)
	ax.set_ylim(-ylims, ylims)
	
	# Save plot to file
	if values.output != None:
		fn = abspath(values.output)
		if '.' not in fn:
			fn += '.png'
		fig.savefig(fn)
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

