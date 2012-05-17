#!/usr/bin/env python2.7

import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib as mplib
import sys
import argparse
from operator import itemgetter
from math import isnan
from galstarutils import *


def main():
	# Parse commandline arguments
	parser = argparse.ArgumentParser(prog='stackpdfs', description='Plot stacked pdfs', add_help=True)
	parser.add_argument('files', nargs='+', type=str, help='Input posterior distributions')
	parser.add_argument('--params', nargs=2, type=str, default=("DM","Ar"), help='Names of parameters, in order (default, "DM Ar")')
	parser.add_argument('--converged', nargs='+', type=str, help='Filter out nonconverged stars using provided stats files')
	parser.add_argument('--output', type=str, required=True, help='Output image filename base (without extension)')
	parser.add_argument('--xmin', type=float, default=None, help='Lower bound of x in plots')
	parser.add_argument('--xmax', type=float, default=None, help='Upper bound of x in plots')
	parser.add_argument('--ymin', type=float, default=None, help='Lower bound of y in plots')
	parser.add_argument('--ymax', type=float, default=None, help='Upper bound of y in plots')
	parser.add_argument('--linear', action='store_true', help='Sum the pdfs linearly')
	parser.add_argument('--norm', action='store_true', help='Normalize the pdf at each distance')
	parser.add_argument('--overplot', type=str, default=None, help='Overplot true values from galfast FITS file')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	# Determine names and indices of parameters
	param_dict = {'dm':(0, '\mu'), 'ar':(1, 'A_r'), 'mr':(2, 'M_r'), 'feh':(3, 'Z')}
	param_indices, param_labels = [], []
	for p in values.params:
		try:
			tmp = param_dict[p.lower()]
			param_indices.append(tmp[0])
			param_labels.append(tmp[1])
		except:
			print 'Invalid parameter name: "%s"' % p
			print 'Valid parameter names are DM, Ar, Mr and FeH.'
			return 1
	
	# Sort filenames
	files = sort_filenames(values.files)
	
	# Load stats files and filter out nonconverged stars
	stats_fn = None
	N = len(files)
	convergence_filter = np.empty(N, dtype=bool)
	convergence_filter.fill(True)
	if values.converged != None:
		stats_fn = sort_filenames(values.converged)
		for i,fn in enumerate(stats_fn):
			converged, mean, cov, ML_dim, ML = read_stats(fn)
			convergence_filter[i] = converged
	files_tmp = []
	for i,f in enumerate(files):
		if convergence_filter[i]:
			files_tmp.append(f)
	files = files_tmp
	N = len(files)
	
	# Get x and y bounds for plot
	xlim = (values.xmin, values.xmax)
	ylim = (values.ymin, values.ymax)
	
	# Load the stacked pdfs
	tmp = load_stacked(files, values.linear, values.norm)
	img, x, y, p = None, None, None, None
	if len(tmp) == 4:
		img, x, y, p = tmp
	
	# Load the true positions of the stars to overlplot
	if values.overplot != None:
		ra_dec, mags, errs, params = get_objects(abspath(values.overplot))
		for i in range(len(params)):
			if params[i,0] > 16.:
				print i
	
	# Determine the output filename
	out_fn = values.output
	if '.png' not in out_fn:
		out_fn += '.png'
	
	# Determine the title and vmin
	figtitle = None
	vmin = None
	if values.norm:
		if values.linear:
			figtitle = r'$\sum_i p_i (' + param_labels[0] + '\, , \,' + param_labels[1] + ') \mathrm{, \ peak \ normalized \ to \ unity \ at \ each \ distance}$'
		else:
			figtitle = r'$\sum_i \ln p_i (' + param_labels[0] + '\, , \,' + param_labels[1] + ') \mathrm{, \ normalized \ and \ stretched \ at \ each \ distance}$'
			vmin = np.max(np.min(img, axis=1))
	else:
		if values.linear:
			figtitle = r'$\sum_i p_i (' + param_labels[0] + '\, , \,' + param_labels[1] + ') \mathrm{, \ peak \ normalized \ to \ unity}$'
		else:
			figtitle = r'$\sum_i \ln p_i (' + param_labels[0] + '\, , \,' + param_labels[1] + ') \mathrm{, \ peak \ normalized \ to \ zero}$'
	
	# Make figure
	print 'Generating plot...'
	mplib.rc('text', usetex=True)
	mplib.rc('xtick', direction='out')
	mplib.rc('ytick', direction='out')
	fig = plt.figure(figsize=(8.5,11.))
	ax = fig.add_subplot(1, 1, 1, axisbg='k')
	plotimg(img, x, y, ax, param_labels, xlim, ylim, vmin=vmin)
	if values.overplot != None:
		x = params[:,0]
		y = params[:,1]
		ax.plot(x, y, 'g.', linestyle='None', markersize=2)
	fig.suptitle(figtitle, y=0.95, fontsize=18)
	fig.savefig(out_fn, transparent=False, dpi=300)
	
	print 'Done.'
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()
